#evaluate.py

"""
Unified quantitative evaluation entry point.

Runs the full validation/test set and reports metrics to stdout + JSON:
  geomtl:       BLEU-1..4, ROUGE-L, CIDEr, METEOR + mIoU
  resnet_lstm / unet3d_lstm: BLEU-1..4, ROUGE-L, CIDEr, METEOR
  seg_ablation: Accuracy, F1 (micro/macro), IoU (macro/micro/per-class)

Usage:
    python evaluate.py geomtl --decoder_mode scratch --checkpoint ... --data_path ... --output_file ...
    python evaluate.py resnet_lstm --checkpoint ... --data_path ... --output_file ...
    python evaluate.py seg_ablation --checkpoint ... --data_path ... --output_file ...
"""

import argparse
import json
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import confusion_matrix

try:
    from pycocoevalcap.bleu.bleu import Bleu
    from pycocoevalcap.rouge.rouge import Rouge
    from pycocoevalcap.cider.cider import Cider
    from pycocoevalcap.meteor.meteor import Meteor
except ImportError:
    print("ERROR: 'pycocoevalcap' not found. Run: pip install pycocoevalcap")
    sys.exit(1)

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.geomtl import GeoMTL, MultiTemporalCropDataset, collate_fn as geomtl_collate_fn
from models.baselines import ResNetLSTM, UNet3DLSTM
from models.seg_ablation import SegOnlyGeoAgent, MultiTemporalCropDatasetSegOnly, collate_fn as seg_collate_fn, evaluate_segmentation


# ============================================================================
# Captioning metrics
# ============================================================================

def score_captions(gts, res, use_meteor=True):
    """Computes BLEU-1..4, ROUGE-L, CIDEr, and optionally METEOR via pycocoevalcap.

    gts / res: {sample_id: ['caption string']}
    METEOR is optional since it can crash/hang on heavily degenerate
    (e.g. hallucinated, repetitive) text; disable with use_meteor=False
    if evaluating an untrained or badly-collapsed checkpoint.
    """
    scorers = [
        (Bleu(4), ["BLEU_1", "BLEU_2", "BLEU_3", "BLEU_4"]),
        (Rouge(), "ROUGE_L"),
        (Cider(), "CIDEr"),
    ]
    if use_meteor:
        try:
            scorers.append((Meteor(), "METEOR"))
        except Exception:
            print("METEOR unavailable (Java missing?). Skipping.")

    final_scores = {}
    for scorer, method in scorers:
        try:
            score, _ = scorer.compute_score(gts, res)
            if isinstance(method, list):
                for m, s in zip(method, score):
                    final_scores[m] = s * 100
            else:
                final_scores[method] = score * 100
        except Exception as e:
            print(f"Error computing {method}: {e}")
            for m in (method if isinstance(method, list) else [method]):
                final_scores[m] = 0.0

    return final_scores


# ============================================================================
# Geo-MTL evaluation (segmentation + captioning)
# ============================================================================

def _evaluate_geomtl(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading GeoMTL ({args.decoder_mode}) from: {args.checkpoint}")

    model = GeoMTL(args.model_weights, decoder_mode=args.decoder_mode).to(device)

    state_dict = torch.load(args.checkpoint, map_location=device, weights_only=False)
    if list(state_dict.keys())[0].startswith("module."):
        state_dict = {k[7:]: v for k, v in state_dict.items()}
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    val_ds = MultiTemporalCropDataset(
        os.path.join(args.data_path, "validation_data.txt"), args.data_path, model.tokenizer, mode="val"
    )
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=4, collate_fn=geomtl_collate_fn)

    num_classes = 14
    conf_matrix = np.zeros((num_classes, num_classes))
    gts_dict, res_dict = {}, {}
    global_idx = 0

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Evaluating"):
            if not batch:
                continue
            img, mask, q_txt, g_prmpt, g_tgt = [b.to(device) for b in batch]
            seg_logits, cap_logits, _, _ = model(img, q_txt, g_prmpt, g_tgt)

            preds_seg = torch.argmax(seg_logits, dim=1).flatten().cpu().numpy()
            targets_seg = mask.flatten().cpu().numpy()
            valid = (targets_seg >= 0) & (targets_seg < num_classes)
            conf_matrix += confusion_matrix(targets_seg[valid], preds_seg[valid], labels=np.arange(num_classes))

            if cap_logits is not None:
                pred_tokens = torch.argmax(cap_logits, dim=-1)
                for i in range(pred_tokens.shape[0]):
                    img_id = str(global_idx)
                    pred_text = model.tokenizer.decode(pred_tokens[i], skip_special_tokens=True)
                    tgt_ids = g_tgt[i]
                    tgt_ids = tgt_ids[tgt_ids != -100]
                    target_text = model.tokenizer.decode(tgt_ids, skip_special_tokens=True)
                    res_dict[img_id] = [pred_text]
                    gts_dict[img_id] = [target_text]
                    global_idx += 1

    intersection = np.diag(conf_matrix)
    union = conf_matrix.sum(axis=1) + conf_matrix.sum(axis=0) - intersection
    iou_per_class = intersection / (union + 1e-10)
    miou = np.nanmean(iou_per_class[1:])  # skip class 0 (no data)

    nlp_metrics = score_captions(gts_dict, res_dict, use_meteor=not args.fast)
    nlp_metrics["mIoU"] = miou

    _print_and_save(nlp_metrics, f"GeoMTL ({args.decoder_mode})", args.output_file, extra_lines=[f"mIoU (13 classes): {miou:.4f}"])


# ============================================================================
# Classical baseline evaluation (captioning only)
# ============================================================================

def _generate_caption_greedy(model, img, max_len=40):
    model.eval()
    with torch.no_grad():
        features = model.encoder(img)
        return model.decoder.sample(features, max_len)


def _evaluate_baseline(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading {args.model} from: {args.checkpoint}")

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
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=4, collate_fn=geomtl_collate_fn)

    gts_dict, res_dict = {}, {}
    global_idx = 0

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Evaluating"):
            if not batch:
                continue
            img, mask, q_txt, g_prmpt, g_tgt = [b.to(device) for b in batch]
            pred_tokens = _generate_caption_greedy(model, img)

            for i in range(pred_tokens.shape[0]):
                img_id = str(global_idx)
                pred_text = tokenizer.decode(pred_tokens[i], skip_special_tokens=True)
                tgt_ids = g_tgt[i]
                tgt_ids = tgt_ids[tgt_ids != -100]
                target_text = tokenizer.decode(tgt_ids, skip_special_tokens=True)
                res_dict[img_id] = [pred_text]
                gts_dict[img_id] = [target_text]
                global_idx += 1

    nlp_metrics = score_captions(gts_dict, res_dict, use_meteor=not args.fast)
    _print_and_save(nlp_metrics, args.model, args.output_file)


# ============================================================================
# Segmentation-ablation evaluation
# ============================================================================

def _evaluate_seg_ablation(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading seg_ablation from: {args.checkpoint}")

    test_ids_file = os.path.join(args.data_path, args.test_ids_filename)
    test_ds = MultiTemporalCropDatasetSegOnly(test_ids_file, args.data_path)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=seg_collate_fn)

    model = SegOnlyGeoAgent(num_seg_classes=args.num_classes, num_frames=3).to(device)
    state_dict = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    metrics = evaluate_segmentation(model, test_loader, device, num_classes=args.num_classes)

    extra_lines = [
        f"Acc (Global) / F1-Micro: {metrics.get('test/Multiclass_Accuracy', 0.0):.4f}",
        f"F1 (Macro):              {metrics.get('test/Multiclass_F1_Score_Macro', 0.0):.4f}",
        f"mIoU (Macro):            {metrics.get('test/Multiclass_Jaccard_Index', 0.0):.4f}",
        f"IoU (Micro):             {metrics.get('test/Multiclass_Jaccard_Index_Micro', 0.0):.4f}",
    ]
    _print_and_save(metrics, "seg_ablation", args.output_file, extra_lines=extra_lines, print_all=False)


# ============================================================================
# Shared reporting
# ============================================================================

def _print_and_save(metrics, label, output_file, extra_lines=None, print_all=True):
    print("\n" + "=" * 60)
    print(f"RESULTS: {label}")
    print("=" * 60)
    if extra_lines:
        for line in extra_lines:
            print(line)
        print("-" * 60)
    if print_all:
        for k, v in metrics.items():
            print(f"{k:<10}: {v:.4f}" if isinstance(v, float) else f"{k:<10}: {v}")
    print("=" * 60)

    with open(output_file, "w") as f:
        json.dump(metrics, f, indent=4)
    print(f"Saved to {output_file}")


# ============================================================================
# CLI
# ============================================================================

def build_parser():
    parser = argparse.ArgumentParser(description="Geo-MTL project: unified evaluation entry point")
    subparsers = parser.add_subparsers(dest="model", required=True)

    p_geomtl = subparsers.add_parser("geomtl")
    p_geomtl.add_argument("--decoder_mode", choices=["pretrained", "scratch"], required=True)
    p_geomtl.add_argument("--data_path", type=str, required=True)
    p_geomtl.add_argument("--model_weights", type=str, required=True)
    p_geomtl.add_argument("--checkpoint", type=str, required=True)
    p_geomtl.add_argument("--output_file", type=str, default="metrics_geomtl.json")
    p_geomtl.add_argument("--batch_size", type=int, default=16)
    p_geomtl.add_argument("--fast", action="store_true", help="Skip METEOR (avoids hangs on degenerate/hallucinated text).")

    for name in ("resnet_lstm", "unet3d_lstm"):
        p_base = subparsers.add_parser(name)
        p_base.add_argument("--data_path", type=str, required=True)
        p_base.add_argument("--checkpoint", type=str, required=True)
        p_base.add_argument("--output_file", type=str, default=f"metrics_{name}.json")
        p_base.add_argument("--batch_size", type=int, default=16)
        p_base.add_argument("--fast", action="store_true")

    p_seg = subparsers.add_parser("seg_ablation")
    p_seg.add_argument("--data_path", type=str, required=True)
    p_seg.add_argument("--checkpoint", type=str, required=True)
    p_seg.add_argument("--output_file", type=str, default="metrics_seg_ablation.json")
    p_seg.add_argument("--test_ids_filename", type=str, default="test_data.txt")
    p_seg.add_argument("--num_classes", type=int, default=14)
    p_seg.add_argument("--batch_size", type=int, default=16)
    p_seg.add_argument("--num_workers", type=int, default=8)

    return parser


def main():
    args = build_parser().parse_args()

    if args.model == "geomtl":
        _evaluate_geomtl(args)
    elif args.model in ("resnet_lstm", "unet3d_lstm"):
        _evaluate_baseline(args)
    elif args.model == "seg_ablation":
        _evaluate_seg_ablation(args)


if __name__ == "__main__":
    main()