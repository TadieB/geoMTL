"""
Segmentation-only ablation: frozen Prithvi-EO-2.0 backbone with a faithful
reproduction of the terratorch UPerNet decoder (ReshapeTokensToImage +
LearnedInterpolateToPyramidal neck, PPM+FPN decoder, segmentation head).

Self-contained by design: does not share decoder code with geomtl.py, so it
stays a faithful reproduction of the external terratorch reference decoder
rather than drifting if geomtl.py's custom UPerNet implementation changes.
"""

import os
import traceback
from abc import ABC, abstractmethod

import numpy as np
import rasterio
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from einops import rearrange
import albumentations as A
from sklearn.metrics import f1_score, jaccard_score, accuracy_score, confusion_matrix

from .prithvi_mae import PrithviViT

NUM_CLASSES = 14


# ============================================================================
# Decoder components (terratorch-equivalent)
# ============================================================================

class ConvModule(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=0,
                 dilation=1, stride=1, inplace=False, transpose=False, scale_factor=None):
        super().__init__()
        conv_name = "ConvTranspose2d" if transpose else "Conv2d"
        if transpose:
            stride = scale_factor
            padding = (kernel_size - scale_factor) // 2
        conv_template = getattr(nn, conv_name)
        self.conv = conv_template(in_channels, out_channels, kernel_size,
                                   padding=padding, dilation=dilation, stride=stride, bias=False)
        self.norm = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU(inplace=inplace)

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))


class PPM(nn.ModuleList):
    """Pyramid pooling module, as used in PSPNet / terratorch's UPerNet decoder."""

    def __init__(self, pool_scales, in_channels, channels, align_corners):
        super().__init__()
        self.align_corners = align_corners
        for pool_scale in pool_scales:
            self.append(nn.Sequential(
                nn.AdaptiveAvgPool2d(pool_scale),
                ConvModule(in_channels, channels, 1, inplace=True),
            ))

    def forward(self, x):
        outs = []
        for ppm in self:
            out = ppm(x)
            outs.append(F.interpolate(out, size=x.size()[2:], mode="bilinear", align_corners=self.align_corners))
        return outs


class UperNetDecoder(nn.Module):
    def __init__(self, embed_dim, pool_scales=(1, 2, 3, 6), channels=512, align_corners=False):
        super().__init__()
        self.embed_dim = embed_dim
        self.out_channels = channels
        self.align_corners = align_corners

        self.psp_modules = PPM(pool_scales, embed_dim[-1], channels, align_corners)
        self.bottleneck = ConvModule(embed_dim[-1] + len(pool_scales) * channels, channels, 3, padding=1, inplace=True)

        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()
        for in_channels in embed_dim[:-1]:
            self.lateral_convs.append(ConvModule(in_channels, channels, 1, inplace=False))
            self.fpn_convs.append(ConvModule(channels, channels, 3, padding=1, inplace=False))

        self.fpn_bottleneck = ConvModule(len(embed_dim) * channels, channels, 3, padding=1, inplace=True)

    def psp_forward(self, inputs):
        x = inputs[-1]
        psp_outs = [x] + self.psp_modules(x)
        return self.bottleneck(torch.cat(psp_outs, dim=1))

    def forward(self, inputs):
        laterals = [conv(inputs[i]) for i, conv in enumerate(self.lateral_convs)]
        laterals.append(self.psp_forward(inputs))

        for i in range(len(laterals) - 1, 0, -1):
            laterals[i - 1] = laterals[i - 1] + F.interpolate(
                laterals[i], size=laterals[i - 1].shape[2:], mode="bilinear", align_corners=self.align_corners
            )

        fpn_outs = [self.fpn_convs[i](laterals[i]) for i in range(len(laterals) - 1)]
        fpn_outs.append(laterals[-1])
        for i in range(len(fpn_outs) - 1, 0, -1):
            fpn_outs[i] = F.interpolate(
                fpn_outs[i], size=fpn_outs[0].shape[2:], mode="bilinear", align_corners=self.align_corners
            )

        return self.fpn_bottleneck(torch.cat(fpn_outs, dim=1))


class SegmentationHead(nn.Module):
    def __init__(self, in_channels, num_classes, channel_list=None, dropout=0):
        super().__init__()
        if channel_list is None:
            pre_head = nn.Identity()
        else:
            def block(c_in, c_out):
                return nn.Sequential(nn.Conv2d(c_in, c_out, kernel_size=3, padding=1), nn.ReLU())
            full_list = [in_channels, *channel_list]
            pre_head = nn.Sequential(*[block(full_list[i], full_list[i + 1]) for i in range(len(full_list) - 1)])
            in_channels = full_list[-1]

        dropout_layer = nn.Identity() if dropout == 0 else nn.Dropout(dropout)
        self.head = nn.Sequential(pre_head, dropout_layer, nn.Conv2d(in_channels, num_classes, kernel_size=1))

    def forward(self, x):
        return self.head(x)


class Neck(ABC, nn.Module):
    def __init__(self, channel_list):
        super().__init__()
        self.channel_list = channel_list

    @abstractmethod
    def process_channel_list(self, channel_list): ...

    @abstractmethod
    def forward(self, channel_list, **kwargs): ...


class ReshapeTokensToImage(Neck):
    """Converts a flat ViT token sequence into a spatial (B, C, H, W) map."""

    def __init__(self, channel_list, remove_cls_token=True, effective_time_dim=1, h=None):
        super().__init__(channel_list)
        self.remove_cls_token = remove_cls_token
        self.effective_time_dim = effective_time_dim
        self.grid_size_h = h if h is not None else 224 // 16
        self.grid_size_w = self.grid_size_h

    def forward(self, features, **kwargs):
        out = []
        for x in features:
            if x.dim() != 3:
                out.append(x)
                continue
            x_no_token = x[:, 1:, :] if self.remove_cls_token else x
            B, N, C = x_no_token.shape
            tokens_per_timestep = N // self.effective_time_dim
            h, w = self.grid_size_h, self.grid_size_w
            if h * w != tokens_per_timestep:
                raise ValueError(f"Cannot infer grid ({h}x{w}) from {tokens_per_timestep} tokens.")
            out.append(rearrange(x_no_token, "b (t h w) c -> b (t c) h w", t=self.effective_time_dim, h=h, w=w))
        return out

    def process_channel_list(self, channel_list):
        return [c * self.effective_time_dim for c in channel_list]


class LearnedInterpolateToPyramidal(Neck):
    """Builds a synthetic multi-scale pyramid from 4 same-resolution feature maps."""

    def __init__(self, channel_list):
        super().__init__(channel_list)
        if len(channel_list) != 4:
            raise ValueError("Requires exactly 4 input embeddings.")
        self.fpn1 = nn.Sequential(
            nn.ConvTranspose2d(channel_list[0], channel_list[0] // 2, 2, 2),
            nn.BatchNorm2d(channel_list[0] // 2), nn.GELU(),
            nn.ConvTranspose2d(channel_list[0] // 2, channel_list[0] // 4, 2, 2),
        )
        self.fpn2 = nn.Sequential(nn.ConvTranspose2d(channel_list[1], channel_list[1] // 2, 2, 2))
        self.fpn3 = nn.Sequential(nn.Identity())
        self.fpn4 = nn.Sequential(nn.MaxPool2d(kernel_size=2, stride=2))

    def forward(self, features, **kwargs):
        return [self.fpn1(features[0]), self.fpn2(features[1]), self.fpn3(features[2]), self.fpn4(features[3])]

    def process_channel_list(self, channel_list):
        return [channel_list[0] // 4, channel_list[1] // 2, channel_list[2], channel_list[3]]


# ============================================================================
# Model
# ============================================================================

class SegOnlyGeoAgent(nn.Module):
    def __init__(self, num_seg_classes, backbone_weights_path=None, num_frames=3, freeze_prithvi=True):
        super().__init__()
        self.num_frames = num_frames

        self.backbone = PrithviViT(
            img_size=224, patch_size=(1, 16, 16), num_frames=num_frames, in_chans=6,
            embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4.0, norm_layer=nn.LayerNorm,
        )
        if backbone_weights_path is not None:
            self._load_backbone_weights(backbone_weights_path)

        self.seg_indices = [5, 11, 17, 23]

        backbone_channels = [self.backbone.embed_dim] * 4
        self.neck_reshape = ReshapeTokensToImage(
            channel_list=backbone_channels, effective_time_dim=num_frames, remove_cls_token=False
        )
        reshaped_channels = [c * num_frames for c in backbone_channels]
        self.neck_pyramid = LearnedInterpolateToPyramidal(channel_list=reshaped_channels)
        pyramid_channels = self.neck_pyramid.process_channel_list(reshaped_channels)

        self.decoder = UperNetDecoder(embed_dim=pyramid_channels, channels=512)
        self.head = SegmentationHead(in_channels=self.decoder.out_channels, num_classes=num_seg_classes, dropout=0.1)

        self.set_prithvi_trainable(not freeze_prithvi)

    def _load_backbone_weights(self, path):
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        encoder_state_dict = {k.replace("encoder.", ""): v for k, v in checkpoint.items() if k.startswith("encoder.")}

        if "pos_embed" in encoder_state_dict:
            pretrained_pos_embed = encoder_state_dict["pos_embed"]
            expected_size = self.backbone.patch_embed.num_patches + 1
            if pretrained_pos_embed.shape[1] != expected_size:
                cls_token_embed = pretrained_pos_embed[:, 0:1, :]
                patch_embeds = pretrained_pos_embed[:, 1:, :]
                embed_dim = pretrained_pos_embed.shape[-1]
                orig_t = (pretrained_pos_embed.shape[1] - 1) // (14 * 14)
                patch_embeds = patch_embeds.reshape(1, orig_t, 14, 14, embed_dim).permute(0, 4, 1, 2, 3)
                patch_embeds = F.interpolate(patch_embeds, size=(self.num_frames, 14, 14), mode="trilinear", align_corners=False)
                patch_embeds = patch_embeds.permute(0, 2, 3, 4, 1).reshape(1, self.num_frames * 14 * 14, embed_dim)
                encoder_state_dict["pos_embed"] = torch.cat([cls_token_embed, patch_embeds], dim=1)

        self.backbone.load_state_dict(encoder_state_dict, strict=False)

    def set_prithvi_trainable(self, trainable=False):
        for param in self.backbone.parameters():
            param.requires_grad = trainable

    def forward(self, x):
        B, T, C, H, W = x.shape
        x = x.permute(0, 2, 1, 3, 4)

        patch_embeds = self.backbone.patch_embed(x)
        pos_embed = self.backbone.interpolate_pos_encoding((T, H, W))
        embeddings = patch_embeds + pos_embed[:, 1:, :]
        cls_tokens = (self.backbone.cls_token + pos_embed[:, :1, :]).expand(B, -1, -1)
        hidden_states = torch.cat((cls_tokens, embeddings), dim=1)

        layer_outputs = []
        for i, blk in enumerate(self.backbone.blocks):
            hidden_states = blk(hidden_states)
            if i in self.seg_indices:
                layer_outputs.append(hidden_states)

        layer_outputs_no_cls = [t[:, 1:, :] for t in layer_outputs]
        reshaped_features = self.neck_reshape(layer_outputs_no_cls)
        pyramid_features = self.neck_pyramid(reshaped_features)
        decoder_output = self.decoder(pyramid_features)
        seg_logits = self.head(decoder_output)

        return F.interpolate(seg_logits, size=(H, W), mode="bilinear", align_corners=False)


# ============================================================================
# Dataset
# ============================================================================

class MultiTemporalCropDatasetSegOnly(Dataset):
    """Loads multi-temporal, multispectral chips with mask targets only.

    Supports train / validation / test splits; test chips are read from the
    validation_chips folder using a separate ID list.
    """

    def __init__(self, chip_ids_file, root_dir, apply_augmentation=False):
        self.root_dir = root_dir
        self.expected_image_shape = (18, 224, 224)
        self.expected_mask_shape = (224, 224)

        with open(chip_ids_file, "r") as f:
            self.chip_ids = [line.strip() for line in f.readlines()]

        if "training" in chip_ids_file:
            self.chip_folder = "training_chips"
            self.apply_augmentation = apply_augmentation
        elif "validation" in chip_ids_file:
            self.chip_folder = "validation_chips"
            self.apply_augmentation = False
        elif "test" in chip_ids_file:
            self.chip_folder = "validation_chips"
            self.apply_augmentation = False
        else:
            raise ValueError(f"Cannot determine split from chip_ids_file name: {chip_ids_file}")

        self.split_data_path = os.path.join(self.root_dir, self.chip_folder)
        self.transform = (
            A.Compose([A.RandomRotate90(p=0.5), A.HorizontalFlip(p=0.5), A.VerticalFlip(p=0.5)])
            if self.apply_augmentation else None
        )

    def __len__(self):
        return len(self.chip_ids)

    def __getitem__(self, idx):
        chip_id = self.chip_ids[idx]
        img_path = os.path.join(self.split_data_path, f"{chip_id}_merged.tif")
        mask_path = os.path.join(self.split_data_path, f"{chip_id}.mask.tif")

        if not (os.path.exists(img_path) and os.path.exists(mask_path)):
            return None

        try:
            with rasterio.open(img_path) as src:
                image = src.read().astype(np.float32)
            if image.shape != self.expected_image_shape:
                return None

            with rasterio.open(mask_path) as src:
                mask = src.read(1).astype(np.int64)
            if mask.shape != self.expected_mask_shape:
                return None

            image = np.transpose(image, (1, 2, 0))
            if self.transform:
                augmented = self.transform(image=image, mask=mask)
                image, mask = augmented["image"], augmented["mask"]

            image = np.transpose(image, (2, 0, 1)).reshape(3, 6, 224, 224)
            return torch.from_numpy(image.copy()), torch.from_numpy(mask.copy())

        except Exception:
            traceback.print_exc()
            return None


def collate_fn(batch):
    batch = [b for b in batch if b is not None]
    if not batch:
        return None
    return torch.utils.data.dataloader.default_collate(batch)


# ============================================================================
# Metrics
# ============================================================================

def evaluate_segmentation(model, dataloader, device, criterion_seg=None, num_classes=NUM_CLASSES):
    """Computes accuracy, F1 (micro/macro), and IoU (macro/micro/per-class) metrics.

    If criterion_seg is provided, also returns average loss under key 'test/loss'.
    """
    model.eval()
    total_loss, batches_processed = 0.0, 0
    all_preds, all_gt = [], []
    class_indices = list(range(num_classes))

    with torch.no_grad():
        for batch_data in dataloader:
            if batch_data is None:
                continue
            images, masks = batch_data
            images = images.to(device, non_blocking=True)
            masks_long = masks.to(device, non_blocking=True).long()

            seg_logits = model(images)

            if criterion_seg is not None:
                loss = criterion_seg(seg_logits, masks_long)
                if not (torch.isnan(loss) or torch.isinf(loss)):
                    total_loss += loss.item()
                    batches_processed += 1

            _, predicted = torch.max(seg_logits, 1)
            all_preds.append(predicted.cpu().numpy().flatten())
            all_gt.append(masks_long.cpu().numpy().flatten())

    metrics = {}
    if criterion_seg is not None:
        metrics["test/loss"] = total_loss / batches_processed if batches_processed > 0 else 0.0

    if not all_preds:
        return metrics

    all_preds_np = np.concatenate(all_preds)
    all_gt_np = np.concatenate(all_gt)

    global_accuracy = accuracy_score(all_gt_np, all_preds_np)
    metrics["test/Multiclass_Accuracy"] = global_accuracy
    metrics["test/Multiclass_F1_Score"] = global_accuracy  # F1-micro == global accuracy

    metrics["test/Multiclass_F1_Score_Macro"] = f1_score(
        all_gt_np, all_preds_np, average="macro", labels=class_indices, zero_division=0
    )

    iou_per_class = jaccard_score(all_gt_np, all_preds_np, average=None, labels=class_indices, zero_division=0)
    metrics["test/Multiclass_Jaccard_Index"] = iou_per_class.mean()
    metrics["test/Multiclass_Jaccard_Index_Micro"] = jaccard_score(
        all_gt_np, all_preds_np, average="micro", labels=class_indices, zero_division=0
    )
    for i, iou in enumerate(iou_per_class):
        metrics[f"test/multiclassjaccardindex_{i}"] = iou

    try:
        matrix = confusion_matrix(all_gt_np, all_preds_np, labels=class_indices)
        TP = matrix.diagonal()
        FP = matrix.sum(axis=0) - TP
        FN = matrix.sum(axis=1) - TP
        TN = matrix.sum() - (TP + FP + FN)
        per_class_accuracy = (TP + TN) / matrix.sum()
        for i, acc in enumerate(per_class_accuracy):
            metrics[f"test/multiclassaccuracy_{i}"] = acc
    except Exception:
        for i in class_indices:
            metrics[f"test/multiclassaccuracy_{i}"] = 0.0

    return metrics