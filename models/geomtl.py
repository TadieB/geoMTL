"""
Geo-MTL model architectures: joint semantic segmentation and scene captioning
on multi-temporal, multispectral Earth Observation imagery.

Two reporting-branch decoder configurations are supported via `decoder_mode`:
  "pretrained": GPT-2 (12 layers), loaded with pretrained weights, frozen.
                Includes an auxiliary image-text contrastive (ITC) loss branch.
  "scratch":    TinyGPT-2 (6 layers), randomly initialized, trained end-to-end.
                No contrastive loss branch.
Q-Former and projector are randomly initialized and trainable in both modes.
"""

import os
import re
import random

import numpy as np
import rasterio
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from einops import rearrange
from transformers import (
    GPT2LMHeadModel, GPT2Tokenizer, GPT2Config,
    Blip2QFormerModel, Blip2QFormerConfig,
)
import albumentations as A

from .prithvi_mae import PrithviViT

NUM_CLASSES = 14
GPT2_VOCAB_SIZE = 50257

CLASS_MAPPING = {
    0: "No Data", 1: "Natural Vegetation", 2: "Forest", 3: "Corn", 4: "Soybeans",
    5: "Wetlands", 6: "Developed/Barren", 7: "Open Water", 8: "Winter Wheat",
    9: "Alfalfa", 10: "Fallow/Idle Cropland", 11: "Cotton", 12: "Sorghum", 13: "Other",
}

HOLISTIC_PROMPTS = [
    "Instruction: Describe this image.",
    "Instruction: Describe this satellite image.",
    "Instruction: Generate a caption for this satellite view.",
    "Instruction: Write a description of the land cover.",
    "Instruction: Analyze the land cover in this image.",
    "Instruction: Explain the agricultural patterns seen here.",
    "Instruction: Provide a detailed report of this area.",
    "Instruction: Summarize the visual content of this image.",
    "Instruction: What does this satellite view show?",
]


# ============================================================================
# Segmentation branch: neck + UPerNet decoder
# ============================================================================

class ConvModule(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=0,
                 dilation=1, stride=1, transpose=False, scale_factor=None):
        super().__init__()
        conv_class = nn.ConvTranspose2d if transpose else nn.Conv2d
        if transpose:
            stride, padding = scale_factor, (kernel_size - scale_factor) // 2
        self.conv = conv_class(in_channels, out_channels, kernel_size,
                                padding=padding, dilation=dilation, stride=stride, bias=False)
        self.norm = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU(inplace=False)

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))


class SpatioTemporalNeck(nn.Module):
    """Reshapes Prithvi's token sequence into a synthetic multi-scale feature pyramid."""

    def __init__(self, embed_dim, frames=3):
        super().__init__()
        self.frames = frames
        c = embed_dim * frames
        self.fpn1 = nn.Sequential(
            nn.ConvTranspose2d(c, c // 2, 2, 2), nn.BatchNorm2d(c // 2), nn.GELU(),
            nn.ConvTranspose2d(c // 2, c // 4, 2, 2),
        )
        self.fpn2 = nn.Sequential(nn.ConvTranspose2d(c, c // 2, 2, 2))
        self.fpn3 = nn.Identity()
        self.fpn4 = nn.Sequential(nn.MaxPool2d(2, 2))
        self.out_channels = [c // 4, c // 2, c, c]

    def forward(self, features):
        reshaped = [rearrange(f, "b (t h w) c -> b (t c) h w", t=self.frames, h=14) for f in features]
        return [self.fpn1(reshaped[0]), self.fpn2(reshaped[1]), self.fpn3(reshaped[2]), self.fpn4(reshaped[3])]


class UPerNetDecoder(nn.Module):
    def __init__(self, embed_dim, channels=512):
        super().__init__()
        self.lateral_convs = nn.ModuleList([ConvModule(d, channels, 1) for d in embed_dim[:-1]])
        self.fpn_convs = nn.ModuleList([ConvModule(channels, channels, 3, padding=1) for _ in embed_dim[:-1]])
        self.fpn_bottleneck = ConvModule(len(embed_dim) * channels, channels, 3, padding=1)
        self.psp = nn.ModuleList([
            nn.Sequential(nn.AdaptiveAvgPool2d(s), ConvModule(embed_dim[-1], channels, 1))
            for s in (1, 2, 3, 6)
        ])
        self.psp_bottleneck = ConvModule(embed_dim[-1] + 4 * channels, channels, 3, padding=1)

    def forward(self, inputs):
        x = inputs[-1]
        psp_outs = [x] + [
            F.interpolate(mod(x), size=x.shape[2:], mode="bilinear", align_corners=False) for mod in self.psp
        ]
        laterals = [conv(inputs[i]) for i, conv in enumerate(self.lateral_convs)]
        laterals.append(self.psp_bottleneck(torch.cat(psp_outs, dim=1)))

        for i in range(len(laterals) - 1, 0, -1):
            laterals[i - 1] = laterals[i - 1] + F.interpolate(
                laterals[i], size=laterals[i - 1].shape[2:], mode="bilinear", align_corners=False
            )

        fpn_outs = [self.fpn_convs[i](laterals[i]) for i in range(len(laterals) - 1)] + [laterals[-1]]
        fpn_outs = [
            F.interpolate(out, size=fpn_outs[0].shape[2:], mode="bilinear", align_corners=False)
            for out in fpn_outs
        ]
        return self.fpn_bottleneck(torch.cat(fpn_outs, dim=1))


# ============================================================================
# Dataset
# ============================================================================

class MultiTemporalCropDataset(Dataset):
    """Loads multi-temporal, multispectral chips with mask and text-report targets.

    Text priority: <chip>.clean.txt > <chip>.qual.txt > <chip>.txt (cleaned on load).
    """

    def __init__(self, chip_ids_file, root_dir, tokenizer, max_len=128, augment=False, mode="train"):
        self.root_dir = root_dir
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.mode = mode

        with open(chip_ids_file, "r") as f:
            self.chip_ids = [line.strip() for line in f.readlines()]
        self.folder = "training_chips" if "training" in chip_ids_file else "validation_chips"
        self.path = os.path.join(root_dir, self.folder)

        self.transform = (
            A.Compose([A.RandomRotate90(p=0.5), A.HorizontalFlip(p=0.5), A.VerticalFlip(p=0.5)])
            if augment else None
        )
        self.holistic_prompts = HOLISTIC_PROMPTS

    def __len__(self):
        return len(self.chip_ids)

    def clean_text_qualitative(self, text):
        """Fallback cleaner: strips statistics and technical jargon."""
        text = re.sub(r"\s*\(\d+(\.\d+)?\s?%\)", "", text)
        text = re.sub(r"\d+(\.\d+)?\s?%", "", text)
        text = re.sub(r"\b\d{4}\b", "", text)
        jargon = ["false-color", "false color", "infrared", "composite", "multispectral", "high-resolution"]
        for word in jargon:
            text = re.sub(rf"\b{word}\b", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s+", " ", text).strip()
        text = text.replace(" ,", ",").replace(" .", ".")
        if len(text) > 0:
            text = text[0].upper() + text[1:]
        return text

    def safe_tokenize(self, text, max_len):
        if not text or text.strip() == "":
            text = "image"
        try:
            out = self.tokenizer(text, return_tensors="pt", padding="max_length", truncation=True, max_length=max_len)
            ids = out["input_ids"][0]
        except Exception:
            ids = torch.full((max_len,), self.tokenizer.pad_token_id, dtype=torch.long)

        if ids.size(0) != max_len:
            curr_len = ids.size(0)
            if curr_len < max_len:
                padding = torch.full((max_len - curr_len,), self.tokenizer.pad_token_id, dtype=torch.long)
                ids = torch.cat([ids, padding])
            else:
                ids = ids[:max_len]

        return torch.clamp(ids, min=0, max=GPT2_VOCAB_SIZE - 1).clone().detach()

    def __getitem__(self, idx):
        chip_id = self.chip_ids[idx]
        try:
            img_path = os.path.join(self.path, f"{chip_id}_merged.tif")
            mask_path = os.path.join(self.path, f"{chip_id}.mask.tif")
            if not os.path.exists(img_path):
                return None

            with rasterio.open(img_path) as src:
                img = src.read().astype(np.float32)
            with rasterio.open(mask_path) as src:
                mask = src.read(1).astype(np.int64)

            mask[mask == 255] = 0
            mask[mask < 0] = 0
            mask = np.clip(mask, 0, 13)

            img = np.transpose(img, (1, 2, 0))
            if self.transform:
                aug = self.transform(image=img, mask=mask)
                img, mask = aug["image"], aug["mask"]
            img = np.transpose(img, (2, 0, 1)).reshape(3, 6, 224, 224).copy()

            clean_path = os.path.join(self.path, f"{chip_id}.clean.txt")
            qual_path = os.path.join(self.path, f"{chip_id}.qual.txt")
            full_path = os.path.join(self.path, f"{chip_id}.txt")

            target_text = ""
            if os.path.exists(clean_path):
                target_text = open(clean_path).read().strip()
            elif os.path.exists(qual_path):
                target_text = self.clean_text_qualitative(open(qual_path).read().strip())
            elif os.path.exists(full_path):
                target_text = self.clean_text_qualitative(open(full_path).read().strip())

            prompt_text = (
                random.choice(self.holistic_prompts) if self.mode == "train"
                else "Instruction: Describe this image."
            )

            q_input_ids = self.safe_tokenize(target_text, 128)
            gpt_prompt_ids = self.safe_tokenize(prompt_text, 128)
            gpt_target_ids = self.safe_tokenize(target_text + self.tokenizer.eos_token, 128)

            return torch.from_numpy(img), torch.from_numpy(mask.copy()), q_input_ids, gpt_prompt_ids, gpt_target_ids

        except Exception:
            return None


def collate_fn(batch):
    batch = [b for b in batch if b is not None]
    if not batch:
        return None
    imgs, masks, q_ids, p_ids, t_ids = zip(*batch)
    return torch.stack(imgs), torch.stack(masks), torch.stack(q_ids), torch.stack(p_ids), torch.stack(t_ids)


# ============================================================================
# Contrastive loss (pretrained-decoder variant only)
# ============================================================================

class ContrastiveLoss(nn.Module):
    def __init__(self, temp=0.07):
        super().__init__()
        self.temp = temp

    def forward(self, vis, txt):
        vis = F.normalize(vis.mean(dim=1), dim=-1)
        txt = F.normalize(txt, dim=-1)
        sim = torch.matmul(vis, txt.t()) / self.temp
        labels = torch.arange(sim.size(0), device=sim.device)
        return (F.cross_entropy(sim, labels) + F.cross_entropy(sim.t(), labels)) / 2


# ============================================================================
# Geo-MTL model
# ============================================================================

class GeoMTL(nn.Module):
    """Joint segmentation and captioning model over a frozen Prithvi-EO-2.0 backbone.

    decoder_mode:
      "pretrained": frozen GPT-2 (12 layers), plus an ITC contrastive loss branch.
      "scratch":    trainable TinyGPT-2 (6 layers), no contrastive loss branch.
    Q-Former and projector are randomly initialized and trainable in both modes.
    """

    def __init__(self, backbone_weights_path, decoder_mode="scratch"):
        super().__init__()
        assert decoder_mode in ("pretrained", "scratch")
        self.decoder_mode = decoder_mode

        self.backbone = PrithviViT(
            img_size=224, patch_size=(1, 16, 16), num_frames=3,
            in_chans=6, embed_dim=1024, depth=24, num_heads=16, norm_layer=nn.LayerNorm,
        )
        self._load_backbone_weights(backbone_weights_path)
        self.seg_indices = [5, 11, 17, 23]

        self.neck = SpatioTemporalNeck(1024)
        self.seg_decoder = UPerNetDecoder(self.neck.out_channels)
        self.seg_head = nn.Conv2d(512, NUM_CLASSES, 1)

        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        if decoder_mode == "pretrained":
            self.gpt2 = GPT2LMHeadModel.from_pretrained("gpt2")
            self.text_embeddings = nn.Embedding(len(self.tokenizer), 768, padding_idx=self.tokenizer.pad_token_id)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=768, nhead=12, dim_feedforward=3072, batch_first=True, norm_first=True
            )
            self.text_encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)
        else:
            tiny_config = GPT2Config(vocab_size=len(self.tokenizer), n_embd=768, n_layer=6, n_head=8, n_positions=512)
            self.gpt2 = GPT2LMHeadModel(tiny_config)
            self.text_embeddings = None
            self.text_encoder = None

        q_config = Blip2QFormerConfig(
            vocab_size=len(self.tokenizer), hidden_size=768, num_hidden_layers=6,
            num_attention_heads=12, encoder_hidden_size=1024, use_qformer_text_input=False,
        )
        self.qformer = Blip2QFormerModel(q_config)
        self.query_tokens = nn.Parameter(torch.randn(1, 32, 768))
        self.projector = nn.Linear(768, self.gpt2.config.n_embd)

        for p in self.backbone.parameters():
            p.requires_grad = False
        if decoder_mode == "pretrained":
            for p in self.gpt2.parameters():
                p.requires_grad = False

    def _load_backbone_weights(self, path):
        if not os.path.exists(path):
            print(f"WARNING: backbone weights not found at {path}")
            return
        checkpoint = torch.load(path, map_location="cpu")
        state_dict = (
            checkpoint if "encoder.pos_embed" not in checkpoint
            else {k.replace("encoder.", ""): v for k, v in checkpoint.items()}
        )
        if "pos_embed" in state_dict:
            pretrained_pos = state_dict["pos_embed"]
            current_num_tokens = self.backbone.patch_embed.num_patches + 1
            if pretrained_pos.shape[1] != current_num_tokens:
                cls_token = pretrained_pos[:, 0:1, :]
                patch_embeds = pretrained_pos[:, 1:, :]
                embed_dim = pretrained_pos.shape[-1]
                t_old = patch_embeds.shape[1] // (14 * 14)
                patch_embeds = patch_embeds.reshape(1, t_old, 14, 14, embed_dim).permute(0, 4, 1, 2, 3)
                patch_embeds = F.interpolate(patch_embeds, size=(3, 14, 14), mode="trilinear", align_corners=False)
                patch_embeds = patch_embeds.permute(0, 2, 3, 4, 1).reshape(1, 3 * 14 * 14, embed_dim)
                state_dict["pos_embed"] = torch.cat([cls_token, patch_embeds], dim=1)
        self.backbone.load_state_dict(state_dict, strict=False)

    def forward(self, img, q_text_ids=None, gpt_prompt_ids=None, gpt_target_ids=None):
        B = img.shape[0]

        x = img.permute(0, 2, 1, 3, 4)
        patch_embeds = self.backbone.patch_embed(x)
        pos_embed = self.backbone.interpolate_pos_encoding((3, 224, 224))
        embeddings = patch_embeds + pos_embed[:, 1:, :]
        cls_token = self.backbone.cls_token + pos_embed[:, :1, :]
        hidden_states = torch.cat((cls_token.expand(B, -1, -1), embeddings), dim=1)

        layer_outputs = []
        for i, blk in enumerate(self.backbone.blocks):
            hidden_states = blk(hidden_states)
            if i in self.seg_indices:
                layer_outputs.append(hidden_states)

        neck_in = [s[:, 1:, :] for s in layer_outputs]
        neck_out = self.neck(neck_in)
        seg_out = self.seg_decoder(neck_out)
        seg_logits = self.seg_head(seg_out)
        seg_logits = F.interpolate(seg_logits, size=(224, 224), mode="bilinear")

        prithvi_patch_features = layer_outputs[-1][:, 1:, :]
        image_atts = torch.ones(prithvi_patch_features.size()[:-1], dtype=torch.long).to(img.device)
        query_tokens = self.query_tokens.expand(B, -1, -1)
        q_out_vis = self.qformer(
            query_embeds=query_tokens,
            encoder_hidden_states=prithvi_patch_features,
            encoder_attention_mask=image_atts,
        )
        grounded_visual_summary = q_out_vis.last_hidden_state[:, :32, :]

        text_feats = None
        if self.decoder_mode == "pretrained" and q_text_ids is not None:
            key_padding_mask = (q_text_ids == self.tokenizer.pad_token_id)
            t_embeds = self.text_embeddings(q_text_ids)
            t_encoded = self.text_encoder(t_embeds, src_key_padding_mask=key_padding_mask)
            mask_float = (~key_padding_mask).float().unsqueeze(-1)
            text_feats = (t_encoded * mask_float).sum(dim=1) / mask_float.sum(dim=1).clamp(min=1e-9)

        visual_prefix = self.projector(grounded_visual_summary)
        cap_logits = None
        if gpt_prompt_ids is not None and gpt_target_ids is not None:
            prompt_embeds = self.gpt2.transformer.wte(gpt_prompt_ids)
            target_embeds = self.gpt2.transformer.wte(gpt_target_ids)
            inputs_embeds = torch.cat([visual_prefix, prompt_embeds, target_embeds], dim=1)
            vis_mask = torch.ones(B, visual_prefix.shape[1], device=img.device)
            p_mask = (gpt_prompt_ids != self.tokenizer.pad_token_id).long()
            t_mask = (gpt_target_ids != self.tokenizer.pad_token_id).long()
            attention_mask = torch.cat([vis_mask, p_mask, t_mask], dim=1)
            outputs = self.gpt2(inputs_embeds=inputs_embeds, attention_mask=attention_mask)
            cap_logits = outputs.logits

        return seg_logits, cap_logits, grounded_visual_summary, text_feats