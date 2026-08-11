#inference.py
"""
Unified qualitative inference: generates captions + (for GeoMTL) segmentation
masks for a handful of validation samples, saving per-sample text reports and
paper-style figures.

Usage:
    python inference.py geomtl --decoder_mode scratch --checkpoint ... --model_weights ... --data_path ... --output_dir ...
    python inference.py geomtl --decoder_mode pretrained --checkpoint ... --model_weights ... --data_path ... --output_dir ...
    python inference.py resnet_lstm --checkpoint ... --data_path ... --output_dir ...
    python inference.py unet3d_lstm --checkpoint ... --data_path ... --output_dir ...
"""

import argparse
import json
import os
import sys
import random
import textwrap

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import BoundaryNorm
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.geomtl import GeoMTL, MultiTemporalCropDataset, CLASS_MAPPING, HOLISTIC_PROMPTS
from models.baselines import ResNetLSTM, UNet3DLSTM

NUM_CLASSES = 14

# ============================================================================
# Shared: RGB extraction, stats, greedy repetition-penalized generation
# ============================================================================

def get_rgb_at_timestep(img_tensor, time_step):
    """Extracts an RGB composite for one timestep from a (T, C, H, W) or
    flattened (T*C, H, W) chip tensor. Assumes the first 3 bands of each
    timestep's 6 bands are RGB."""
    img_tensor = img_tensor.cpu()
    if img_tensor.ndim == 5:
        B, T, C, H, W = img_tensor.shape
        img_tensor = img_tensor.view(B, T * C, H, W).squeeze(0)
    elif img_tensor.ndim == 4:
        img_tensor = img_tensor.squeeze(0)
    if img_tensor.ndim == 3 and img_tensor.shape[0] == 3:
        img_tensor = img_tensor.view(-1, img_tensor.shape[-2], img_tensor.shape[-1])

    start = time_step * 6
    rgb = img_tensor[start:start + 3, :, :].numpy()
    rgb = np.transpose(rgb, (1, 2, 0))
    return (rgb - rgb.min()) / (rgb.max() - rgb.min() + 1e-9)


def calculate_stats_from_mask(mask_tensor):
    """Formats a per-class pixel-percentage breakdown of a predicted mask,
    for inclusion in the saved text report only (does not condition generation)."""
    mask_np = mask_tensor.cpu().numpy()
    unique, counts = np.unique(mask_np, return_counts=True)
    pixel_counts = dict(zip(unique, counts))
    total_valid = sum(c for i, c in pixel_counts.items() if i > 0 and i != 255)
    if total_valid == 0:
        return "No valid crops detected."

    items = []
    for cls_idx, count in sorted(pixel_counts.items(), key=lambda x: x[1], reverse=True):
        if cls_idx <= 0 or cls_idx == 255:
            continue
        pct = (count / total_valid) * 100
        items.append(f"{CLASS_MAPPING.get(cls_idx, 'Unknown')}: {pct:.1f}%")
    return ", ".join(items) + "."

def generate_caption_geomtl(model, img_tensor, prompt, device, repetition_penalty=1.5, max_new_tokens=60):
    """Blind (pure-vision) greedy generation: visual queries + prompt condition
    GPT-2, with a per-token repetition penalty to suppress degenerate loops.
    Applied with the same penalty value across all decoder modes for a fair
    comparison; greedy argmax decoding makes temperature scaling a no-op, so
    no temperature parameter is exposed."""
    model.eval()
    prompt_ids = model.tokenizer(prompt, return_tensors="pt")["input_ids"][0].to(device)

    with torch.no_grad():
        B = img_tensor.shape[0]
        if img_tensor.ndim == 4 and img_tensor.shape[1] == 18:
            _, _, H, W = img_tensor.shape
            img_tensor = img_tensor.view(B, 3, 6, H, W)

        x = img_tensor.permute(0, 2, 1, 3, 4)
        patch_embeds = model.backbone.patch_embed(x)
        pos_embed = model.backbone.interpolate_pos_encoding((3, 224, 224))
        embeddings = patch_embeds + pos_embed[:, 1:, :]
        cls_token = model.backbone.cls_token + pos_embed[:, :1, :]
        hidden_states = torch.cat((cls_token.expand(B, -1, -1), embeddings), dim=1)

        layer_outputs = []
        for i, blk in enumerate(model.backbone.blocks):
            hidden_states = blk(hidden_states)
            if i in model.seg_indices:
                layer_outputs.append(hidden_states)

        prithvi_feats = layer_outputs[-1][:, 1:, :]
        img_atts = torch.ones(prithvi_feats.size()[:-1], dtype=torch.long).to(device)
        query_tokens = model.query_tokens.expand(B, -1, -1)
        q_out = model.qformer(
            query_embeds=query_tokens,
            encoder_hidden_states=prithvi_feats,
            encoder_attention_mask=img_atts,
        )
        visual_summary = q_out.last_hidden_state[:, :32, :]

        visual_embeds = model.projector(visual_summary)
        prompt_embeds = model.gpt2.transformer.wte(prompt_ids)
        inputs_embeds = torch.cat([visual_embeds, prompt_embeds.unsqueeze(0)], dim=1)
        curr_ids = prompt_ids.clone().unsqueeze(0)

        for _ in range(max_new_tokens):
            outputs = model.gpt2(inputs_embeds=inputs_embeds)
            next_token_logits = outputs.logits[:, -1, :]

            for i in range(B):
                for token_id in curr_ids[i]:
                    if next_token_logits[i, token_id] > 0:
                        next_token_logits[i, token_id] /= repetition_penalty
                    else:
                        next_token_logits[i, token_id] *= repetition_penalty

            next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(1)
            curr_ids = torch.cat([curr_ids, next_token], dim=1)
            if next_token.item() == model.tokenizer.eos_token_id:
                break
            next_embed = model.gpt2.transformer.wte(next_token)
            inputs_embeds = torch.cat([inputs_embeds, next_embed], dim=1)

    full_text = model.tokenizer.decode(curr_ids[0], skip_special_tokens=True)
    return full_text.replace(prompt, "").strip()

# ============================================================================
# Qualitative panel: T0 | T1 | T2 | Ground Truth | Prediction
# ============================================================================

def save_qualitative_panel(samples, output_path, class_mapping):
    num_samples = len(samples)
    if num_samples == 0:
        return

    fig, axes = plt.subplots(num_samples, 5, figsize=(15, 3.5 * num_samples))
    if num_samples == 1:
        axes = np.expand_dims(axes, 0)

    cmap = plt.get_cmap("tab20", NUM_CLASSES)
    norm = BoundaryNorm(np.arange(-0.5, NUM_CLASSES + 0.5, 1), cmap.N)
    plt.subplots_adjust(wspace=0.05, hspace=0.05)

    titles = ["T0 (Input)", "T1 (Input)", "T2 (Input)", "Ground Truth Mask", "Predicted Mask (Ours)"]

    for i, (img, pred, gt, _chip_id) in enumerate(samples):
        for t in range(3):
            axes[i, t].imshow(get_rgb_at_timestep(img, t))
            if i == 0:
                axes[i, t].set_title(titles[t], fontsize=14)
        axes[i, 3].imshow(gt.cpu().numpy(), cmap=cmap, norm=norm, interpolation="nearest")
        if i == 0:
            axes[i, 3].set_title(titles[3], fontsize=14)
        axes[i, 4].imshow(pred.cpu().numpy(), cmap=cmap, norm=norm, interpolation="nearest")
        if i == 0:
            axes[i, 4].set_title(titles[4], fontsize=14)

        for ax in axes[i]:
            ax.set_xticks([])
            ax.set_yticks([])
            ax.axis("off")

    patches = [mpatches.Patch(color=cmap(cid), label=class_mapping.get(cid, str(cid))) for cid in range(NUM_CLASSES)]
    fig.legend(handles=patches, loc="lower center", ncol=7, fontsize=11, bbox_to_anchor=(0.5, 0.01), frameon=False)
    plt.subplots_adjust(bottom=0.08, top=0.95, left=0.01, right=0.99)

    plt.savefig(output_path, bbox_inches="tight", dpi=600)
    plt.close()
    print(f"Saved qualitative panel to {output_path}")


# ============================================================================
# Geo-MTL inference pipeline
# ============================================================================
def _run_geomtl(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)
    random.seed(args.seed)

    model = GeoMTL(args.model_weights, decoder_mode=args.decoder_mode).to(device)
    state_dict = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    val_ds = MultiTemporalCropDataset(
        os.path.join(args.data_path, "validation_data.txt"), args.data_path, model.tokenizer, mode="val"
    )
    indices = random.sample(range(len(val_ds)), min(args.num_samples, len(val_ds)))
    panel_samples = []

    for idx in indices:
        chip_id = val_ds.chip_ids[idx]
        print(f"Processing {chip_id}...")

        img, mask, _, _, g_tgt = val_ds[idx]
        img_tensor = img.unsqueeze(0).to(device)

        with torch.no_grad():
            seg_logits, _, _, _ = model(img_tensor)
            pred_mask = torch.argmax(seg_logits, dim=1).squeeze(0)

        predicted_stats_str = calculate_stats_from_mask(pred_mask)
        selected_prompt = random.choice(HOLISTIC_PROMPTS)
        answer = generate_caption_geomtl(
            model, img_tensor, selected_prompt, device,
            repetition_penalty=args.repetition_penalty,
        )

        gt_report = model.tokenizer.decode(g_tgt, skip_special_tokens=True)
        report = (
            f"CHIP ID: {chip_id}\nSAMPLE INDEX: {idx}\nDECODER MODE: {args.decoder_mode}\n{'=' * 60}\n"
            f"PREDICTED STATS: {textwrap.fill(predicted_stats_str, 100)}\n\n"
            f"PROMPT USED: {selected_prompt}\n"
            f"PREDICTION: {textwrap.fill(answer, 100)}\n\n"
            f"REFERENCE: {textwrap.fill(gt_report, 100)}\n{'=' * 60}\n"
        )
        # with open(os.path.join(args.output_dir, f"{chip_id}_report.txt"), "w") as f:
        #     f.write(report)
        with open(os.path.join(args.output_dir, f"{chip_id}_{args.decoder_mode}_report.txt"), "w") as f:
            f.write(report)

        panel_samples.append((img_tensor.cpu(), pred_mask.cpu(), mask.cpu(), chip_id))

    if panel_samples:
        panel_path = os.path.join(args.output_dir, f"qualitative_panel_{args.decoder_mode}.pdf")
        save_qualitative_panel(panel_samples, panel_path, CLASS_MAPPING)

# ============================================================================
# Classical baseline inference pipeline (captioning only, no segmentation)
# ============================================================================
def _run_baseline(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)
    random.seed(args.seed)

    from transformers import GPT2Tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if args.model == "resnet_lstm":
        model = ResNetLSTM(embed_size=768, hidden_size=768, vocab_size=len(tokenizer), pretrained=False).to(device)
    else:
        model = UNet3DLSTM(embed_size=768, hidden_size=768, vocab_size=len(tokenizer)).to(device)

    state_dict = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    val_ds = MultiTemporalCropDataset(
        os.path.join(args.data_path, "validation_data.txt"), args.data_path, tokenizer, mode="val"
    )

    bleu_scores = []
    indices = random.sample(range(len(val_ds)), min(args.num_samples, len(val_ds)))

    for saved, idx in enumerate(indices):
        img, _, _, _, g_tgt = val_ds[idx]
        img_tensor = img.unsqueeze(0).to(device)

        with torch.no_grad():
            features = model.encoder(img_tensor)
            pred_ids = model.decoder.sample_with_penalty(
                features, max_len=50, repetition_penalty=args.repetition_penalty
            )

        pred_text = tokenizer.decode(pred_ids[0], skip_special_tokens=True)
        gt_text = tokenizer.decode(g_tgt, skip_special_tokens=True)

        score = sentence_bleu(
            [gt_text.lower().split()], pred_text.lower().split(),
            smoothing_function=SmoothingFunction().method1,
        )
        bleu_scores.append(score)

        chip_id = f"{args.model}_sample_{saved}"
        report = (
            f"CHIP ID: {chip_id}\nBASELINE MODEL: {args.model}\n{'=' * 60}\n"
            f"PREDICTION: {textwrap.fill(pred_text, 80)}\n\n"
            f"GROUND TRUTH: {textwrap.fill(gt_text, 80)}\n{'=' * 60}\n"
        )
        with open(os.path.join(args.output_dir, f"{chip_id}_report.txt"), "w") as f:
            f.write(report)

        plt.figure(figsize=(6, 6))
        plt.imshow(get_rgb_at_timestep(img.unsqueeze(0), 1))
        plt.axis("off")
        plt.title(f"{args.model} input (middle frame)", fontsize=14, fontweight="bold")
        plt.savefig(os.path.join(args.output_dir, f"{chip_id}_fig.pdf"), bbox_inches="tight", format="pdf")
        plt.savefig(os.path.join(args.output_dir, f"{chip_id}_fig.png"), bbox_inches="tight", dpi=600)
        plt.close()

    avg_bleu = np.mean(bleu_scores) * 100 if bleu_scores else 0.0
    print(f"\n{args.model} sampled BLEU-4 (sentence-level, {len(bleu_scores)} samples): {avg_bleu:.2f}")
    print("Note: for the authoritative corpus-level metric used in the paper, see evaluate.py.")

    with open(os.path.join(args.output_dir, f"{args.model}_sampled_metrics.json"), "w") as f:
        json.dump({"sampled_bleu4": avg_bleu, "n_samples": len(bleu_scores)}, f)


# ============================================================================
# CLI
# ============================================================================

def build_parser():
    parser = argparse.ArgumentParser(description="Geo-MTL project: unified qualitative inference entry point")
    subparsers = parser.add_subparsers(dest="model", required=True)

    p_geomtl = subparsers.add_parser("geomtl")
    p_geomtl.add_argument("--decoder_mode", choices=["pretrained", "scratch"], required=True)
    p_geomtl.add_argument("--data_path", type=str, required=True)
    p_geomtl.add_argument("--model_weights", type=str, required=True)
    p_geomtl.add_argument("--checkpoint", type=str, required=True)
    p_geomtl.add_argument("--output_dir", type=str, default="./inference_results")
    p_geomtl.add_argument("--num_samples", type=int, default=5)
    p_geomtl.add_argument("--repetition_penalty", type=float, default=1.5)
    p_geomtl.add_argument("--seed", type=int, default=42)

    for name in ("resnet_lstm", "unet3d_lstm"):
        p_base = subparsers.add_parser(name)
        p_base.add_argument("--data_path", type=str, required=True)
        p_base.add_argument("--checkpoint", type=str, required=True)
        p_base.add_argument("--output_dir", type=str, default=f"./inference_results_{name}")
        p_base.add_argument("--num_samples", type=int, default=5)
        p_base.add_argument("--repetition_penalty", type=float, default=1.5)
        p_base.add_argument("--seed", type=int, default=42)

    return parser

def main():
    args = build_parser().parse_args()
    if args.model == "geomtl":
        _run_geomtl(args)
    else:
        _run_baseline(args)


if __name__ == "__main__":
    main()
