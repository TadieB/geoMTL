"""
Unified training entry point for all Geo-MTL project models.

Usage:
    python train.py geomtl --decoder_mode scratch ...
    python train.py geomtl --decoder_mode pretrained --itc_weight 1.0 ...
    python train.py resnet_lstm ...
    python train.py unet3d_lstm ...
    python train.py seg_ablation ...

See README.md for the full argument reference per model.
"""

import argparse
import os
import sys

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.geomtl import GeoMTL, MultiTemporalCropDataset, collate_fn as geomtl_collate_fn, ContrastiveLoss
from models.baselines import ResNetLSTM, UNet3DLSTM
from models.seg_ablation import SegOnlyGeoAgent, MultiTemporalCropDatasetSegOnly, collate_fn as seg_collate_fn, evaluate_segmentation
from models.utils import get_warmup_cosine_scheduler

SEG_CLASS_WEIGHTS = [
    0.0, 0.386375, 0.661126, 0.548184, 0.640482, 0.876862, 0.925186,
    3.249462, 1.542289, 2.175141, 2.272419, 3.062762, 3.626097, 1.198702,
]


# ============================================================================
# Geo-MTL (dual-loss: segmentation + captioning [+ ITC for pretrained mode])
# ============================================================================

def _train_geomtl(args):
    device = torch.device("cuda")
    model = GeoMTL(args.model_weights, decoder_mode=args.decoder_mode).to(device)

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Decoder mode: {args.decoder_mode} | Trainable params: {trainable_params:,}")

    train_ds = MultiTemporalCropDataset(
        os.path.join(args.data_path, "training_data.txt"), args.data_path, model.tokenizer, augment=True, mode="train"
    )
    val_ds = MultiTemporalCropDataset(
        os.path.join(args.data_path, "validation_data.txt"), args.data_path, model.tokenizer, augment=False, mode="val"
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4, collate_fn=geomtl_collate_fn)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=4, collate_fn=geomtl_collate_fn)

    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.learning_rate)
    scheduler = get_warmup_cosine_scheduler(optimizer, len(train_loader) * args.warmup_epochs, len(train_loader) * args.epochs)

    class_weights = torch.tensor(SEG_CLASS_WEIGHTS).to(device)
    crit_seg = nn.CrossEntropyLoss(weight=class_weights, ignore_index=-100)
    crit_cap = nn.CrossEntropyLoss(ignore_index=-100)
    crit_itc = ContrastiveLoss() if args.decoder_mode == "pretrained" else None

    best_miou = 0.0

    for epoch in range(args.epochs):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}")

        for batch in pbar:
            if not batch:
                continue
            try:
                img, mask, q_txt, g_prmpt, g_tgt = [b.to(device) for b in batch]
                mask[mask == -1] = -100

                seg_logits, cap_logits, vis_embeds, txt_embeds = model(img, q_txt, g_prmpt, g_tgt)
                l_seg = crit_seg(seg_logits, mask.long())

                L_vis = vis_embeds.shape[1]
                L_prmpt = g_prmpt.shape[1]
                labels = torch.full((img.shape[0], L_vis + L_prmpt + g_tgt.shape[1]), -100, dtype=torch.long, device=device)
                labels[:, L_vis + L_prmpt:] = g_tgt
                labels[labels == model.tokenizer.pad_token_id] = -100

                shift_logits = cap_logits[:, :-1, :].contiguous()
                shift_labels = labels[:, 1:].contiguous()
                l_cap = crit_cap(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

                if crit_itc is not None:
                    l_itc = crit_itc(vis_embeds, txt_embeds)
                    loss = l_seg + l_cap + (args.itc_weight * l_itc)
                    pbar.set_postfix(L_seg=f"{l_seg:.3f}", L_cap=f"{l_cap:.3f}", L_itc=f"{l_itc:.3f}")
                else:
                    loss = l_seg + l_cap
                    pbar.set_postfix(L_seg=f"{l_seg:.3f}", L_cap=f"{l_cap:.3f}")

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()

            except RuntimeError as e:
                if "CUDA" in str(e):
                    print(f"CUDA error at batch, skipping: {e}")
                    torch.cuda.empty_cache()
                    continue
                raise

        v_miou = _validate_geomtl_seg(model, val_loader, crit_seg, device)
        print(f"Epoch {epoch + 1}: val mIoU={v_miou:.4f}")

        if v_miou > best_miou:
            best_miou = v_miou
            ckpt_name = f"geomtl_{args.decoder_mode}_best.pth"
            torch.save(model.state_dict(), os.path.join(args.checkpoint_dir, ckpt_name))


def _validate_geomtl_seg(model, dataloader, criterion_seg, device):
    model.eval()
    from sklearn.metrics import jaccard_score
    import numpy as np

    all_preds, all_masks = [], []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validating", leave=False):
            if not batch:
                continue
            img, mask, _, _, _ = [b.to(device) for b in batch]
            seg_logits, _, _, _ = model(img)
            _, preds = torch.max(seg_logits, 1)
            valid = mask.cpu().numpy().flatten() >= 0
            all_preds.append(preds.cpu().numpy().flatten()[valid])
            all_masks.append(mask.cpu().numpy().flatten()[valid])

    if not all_preds:
        return 0.0
    return jaccard_score(
        np.concatenate(all_masks), np.concatenate(all_preds),
        average="macro", labels=list(range(14)), zero_division=0,
    )


# ============================================================================
# Classical baselines (ResNet-LSTM, 3D-UNet-LSTM): captioning-only, single loss
# ============================================================================

def _train_baseline(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    from transformers import GPT2Tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_ds = MultiTemporalCropDataset(
        os.path.join(args.data_path, "training_data.txt"), args.data_path, tokenizer, augment=True, mode="train"
    )
    val_ds = MultiTemporalCropDataset(
        os.path.join(args.data_path, "validation_data.txt"), args.data_path, tokenizer, augment=False, mode="val"
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4, collate_fn=geomtl_collate_fn)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=4, collate_fn=geomtl_collate_fn)

    if args.model == "resnet_lstm":
        model = ResNetLSTM(embed_size=768, hidden_size=768, vocab_size=len(tokenizer)).to(device)
        ckpt_name = "baseline_resnet_lstm.pth"
    else:
        model = UNet3DLSTM(embed_size=768, hidden_size=768, vocab_size=len(tokenizer)).to(device)
        ckpt_name = "baseline_3d_unet_lstm.pth"

    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    best_loss = float("inf")

    for epoch in range(args.epochs):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}")
        for batch in pbar:
            if not batch:
                continue
            img, _, _, _, target_ids = [b.to(device) for b in batch]
            outputs = model(img, target_ids)
            loss = criterion(outputs.reshape(-1, outputs.shape[2]), target_ids.reshape(-1))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                if not batch:
                    continue
                img, _, _, _, target_ids = [b.to(device) for b in batch]
                outputs = model(img, target_ids)
                val_loss += criterion(outputs.reshape(-1, outputs.shape[2]), target_ids.reshape(-1)).item()

        avg_val = val_loss / len(val_loader)
        print(f"Epoch {epoch + 1} val loss: {avg_val:.4f}")

        if avg_val < best_loss:
            best_loss = avg_val
            torch.save(model.state_dict(), os.path.join(args.checkpoint_dir, ckpt_name))


# ============================================================================
# Segmentation-only ablation
# ============================================================================

def _train_seg_ablation(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_ds = MultiTemporalCropDatasetSegOnly(
        os.path.join(args.data_path, "training_data.txt"), args.data_path, apply_augmentation=True
    )
    val_ds = MultiTemporalCropDatasetSegOnly(
        os.path.join(args.data_path, "validation_data.txt"), args.data_path, apply_augmentation=False
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=seg_collate_fn)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=seg_collate_fn)

    model = SegOnlyGeoAgent(
        num_seg_classes=args.num_classes, backbone_weights_path=args.model_weights,
        num_frames=3, freeze_prithvi=not args.finetune_full_model,
    ).to(device)

    class_weights = torch.tensor([0.0] + SEG_CLASS_WEIGHTS[1:]).to(device)
    criterion_seg = nn.CrossEntropyLoss(weight=class_weights, ignore_index=0)

    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = get_warmup_cosine_scheduler(optimizer, len(train_loader) * args.warmup_epochs, len(train_loader) * args.epochs)

    best_miou = 0.0
    ckpt_path = os.path.join(args.checkpoint_dir, "seg_ablation_best.pth")

    for epoch in range(args.epochs):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}")
        for batch in pbar:
            if batch is None:
                continue
            images, masks = [b.to(device) for b in batch]
            optimizer.zero_grad(set_to_none=True)

            seg_logits = model(images)
            loss = criterion_seg(seg_logits, masks.long())
            if torch.isnan(loss) or torch.isinf(loss):
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
            optimizer.step()
            scheduler.step()
            pbar.set_postfix(L_seg=f"{loss.item():.4f}")

        val_metrics = evaluate_segmentation(model, val_loader, device, criterion_seg)
        current_miou = val_metrics.get("test/Multiclass_Jaccard_Index", 0.0)
        print(f"Epoch {epoch + 1}: val mIoU={current_miou:.4f}")

        if current_miou > best_miou:
            best_miou = current_miou
            torch.save(model.state_dict(), ckpt_path)


# ============================================================================
# CLI
# ============================================================================

def build_parser():
    parser = argparse.ArgumentParser(description="Geo-MTL project: unified training entry point")
    subparsers = parser.add_subparsers(dest="model", required=True)

    p_geomtl = subparsers.add_parser("geomtl", help="Joint segmentation + captioning model")
    p_geomtl.add_argument("--decoder_mode", choices=["pretrained", "scratch"], required=True)
    p_geomtl.add_argument("--itc_weight", type=float, default=1.0,
                           help="Weight for the ITC contrastive loss term (pretrained mode only).")
    p_geomtl.add_argument("--data_path", type=str, required=True)
    p_geomtl.add_argument("--model_weights", type=str, required=True, help="Path to Prithvi-EO-2.0 backbone weights.")
    p_geomtl.add_argument("--checkpoint_dir", type=str, default="./checkpoints/geomtl")
    p_geomtl.add_argument("--epochs", type=int, default=50)
    p_geomtl.add_argument("--batch_size", type=int, default=16)
    p_geomtl.add_argument("--learning_rate", type=float, default=2e-4)
    p_geomtl.add_argument("--warmup_epochs", type=int, default=5)

    for name in ("resnet_lstm", "unet3d_lstm"):
        p_base = subparsers.add_parser(name, help="Classical CNN/3DCNN + LSTM captioning baseline")
        p_base.add_argument("--data_path", type=str, required=True)
        p_base.add_argument("--checkpoint_dir", type=str, default="./checkpoints/baselines")
        p_base.add_argument("--epochs", type=int, default=50)
        p_base.add_argument("--batch_size", type=int, default=16)
        p_base.add_argument("--learning_rate", type=float, default=4e-4)

    p_seg = subparsers.add_parser("seg_ablation", help="Segmentation-only backbone ablation")
    p_seg.add_argument("--data_path", type=str, required=True)
    p_seg.add_argument("--model_weights", type=str, required=True)
    p_seg.add_argument("--checkpoint_dir", type=str, default="./checkpoints/seg_ablation")
    p_seg.add_argument("--num_classes", type=int, default=14)
    p_seg.add_argument("--batch_size", type=int, default=16)
    p_seg.add_argument("--num_workers", type=int, default=8)
    p_seg.add_argument("--weight_decay", type=float, default=0.1)
    p_seg.add_argument("--clip_grad_norm", type=float, default=1.0)
    p_seg.add_argument("--epochs", type=int, default=50)
    p_seg.add_argument("--learning_rate", type=float, default=2e-4)
    p_seg.add_argument("--warmup_epochs", type=int, default=5)
    p_seg.add_argument("--finetune_full_model", action="store_true", help="Unfreeze backbone (full fine-tuning).")

    return parser


def main():
    args = build_parser().parse_args()
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    if args.model == "geomtl":
        _train_geomtl(args)
    elif args.model in ("resnet_lstm", "unet3d_lstm"):
        _train_baseline(args)
    elif args.model == "seg_ablation":
        _train_seg_ablation(args)


if __name__ == "__main__":
    main()