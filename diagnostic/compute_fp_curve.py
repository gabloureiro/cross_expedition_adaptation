#!/usr/bin/env python3
"""
compute_fp_curve.py — Read raw predictions, compute FP rate vs border distance, and plot Figure 4.
"""

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    plt.style.use("seaborn-v0_8-whitegrid")
except OSError:
    try: plt.style.use("seaborn-whitegrid")
    except OSError: pass

TILE_SIZE = 640

# ── Geometry helpers ──────────────────────────────────────────────────────────
def yolo_norm_to_xyxy(xc: float, yc: float, w: float, h: float, s: int = TILE_SIZE) -> List[float]:
    return [(xc - w / 2) * s, (yc - h / 2) * s, (xc + w / 2) * s, (yc + h / 2) * s]

def xywh_to_xyxy(xc: float, yc: float, w: float, h: float) -> List[float]:
    return [xc - w / 2, yc - h / 2, xc + w / 2, yc + h / 2]

def border_dist(box: List[float], tile_size: int = TILE_SIZE) -> float:
    return min(box[0], box[1], tile_size - box[2], tile_size - box[3])

def is_contained(pred_xyxy: List[float], gt_boxes: List[List[float]]) -> bool:
    px, py = (pred_xyxy[0] + pred_xyxy[2]) / 2, (pred_xyxy[1] + pred_xyxy[3]) / 2
    return any(g[0] <= px <= g[2] and g[1] <= py <= g[3] for g in gt_boxes)

# ── CLI ───────────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compute False Positive curve and plot Figure 4.")
    p.add_argument("--raw_preds_dir",   required=True,  type=Path)
    p.add_argument("--labels_dir",      required=True,  type=Path)
    p.add_argument("--output_dir",      required=True,  type=Path)
    p.add_argument("--models",          nargs="+", default=["YOLOv8", "DINO", "FasterRCNN"])
    p.add_argument("--dist_bin_width",  type=int, default=8)
    p.add_argument("--max_dist",        type=int, default=200)
    p.add_argument("--min_preds_per_bin",type=int, default=5)
    return p.parse_args()

# ── Evaluation ────────────────────────────────────────────────────────────────
def compute_curve(model_dir: Path, labels_dir: Path, nonempty_stems: set, bin_width: int, max_dist: int, min_preds: int) -> Dict:
    bins = defaultdict(lambda: {"tp": 0, "fp": 0})
    
    for jf in model_dir.glob("*.json"):
        if jf.stem not in nonempty_stems: continue
        gt_path = labels_dir / f"{jf.stem}.txt"
        
        try:
            data = json.loads(jf.read_text(encoding="utf-8"))
            preds = data if isinstance(data, list) else data.get("boxes", [])
        except Exception: continue

        gt_boxes = [yolo_norm_to_xyxy(*list(map(float, line.split()))[1:5]) 
                    for line in gt_path.read_text().splitlines() if line.strip()]

        for pred in preds:
            box = xywh_to_xyxy(*pred["bbox"])
            bd = border_dist(box)
            if bd > max_dist: continue
            
            bucket = int(math.floor(bd / bin_width)) * bin_width
            if is_contained(box, gt_boxes):
                bins[bucket]["tp"] += 1
            else:
                bins[bucket]["fp"] += 1

    distances, fp_rates = [], []
    for bucket in sorted(bins):
        tp, fp = bins[bucket]["tp"], bins[bucket]["fp"]
        total = tp + fp
        if total >= min_preds:
            distances.append(bucket)
            fp_rates.append(fp / total)
            
    return {"distances": distances, "fp_rates": fp_rates}

# ── Plotting ──────────────────────────────────────────────────────────────────
def plot_curves(data: Dict, output_dir: Path):
    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=150)
    styles = {
        "YOLOv8":     {"color": "#e74c3c", "ls": "-",  "marker": "o"},
        "DINO":       {"color": "#2980b9", "ls": "--", "marker": "s"},
        "FasterRCNN": {"color": "#27ae60", "ls": "-.", "marker": "^"}
    }
    default_colors = ["#8e44ad", "#f39c12", "#16a085"]

    for i, (model_name, curve) in enumerate(data.items()):
        if not curve["distances"]: continue
        s = styles.get(model_name, {"color": default_colors[i % 3], "ls": "-", "marker": "d"})
        fp_pct = [r * 100 for r in curve["fp_rates"]]
        
        ax.plot(curve["distances"], fp_pct, color=s["color"], linestyle=s["ls"], 
                linewidth=2, marker=s["marker"], markersize=4, label=model_name, zorder=3)

    ax.axvline(64, color="black", linestyle=":", linewidth=1.5, alpha=0.7, zorder=4)
    ax.text(67, 95, "64px\nthreshold", fontsize=8, ha="left", va="top", color="black", zorder=5)

    ax.set_xlabel("Distance from tile border (px)", fontsize=11)
    ax.set_ylabel("False Positive Rate (%)", fontsize=11)
    ax.set_xlim(0, 200)
    ax.set_ylim(0, 105)
    ax.set_xticks([0, 32, 64, 96, 128, 160, 192])
    ax.set_yticks([0, 20, 40, 60, 80, 100])

    ax.grid(True, alpha=0.3, linestyle="--")
    ax.legend(loc="upper right", framealpha=0.9, fontsize=10)

    plt.tight_layout()
    fig.savefig(output_dir / "figure4_border_fp_curve.pdf", dpi=300, bbox_inches="tight")
    fig.savefig(output_dir / "figure4_border_fp_curve.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    nonempty_stems = {f.stem for f in args.labels_dir.glob("*.txt") if f.read_text(encoding="utf-8").strip()}
    
    results = {}
    for model_name in args.models:
        model_dir = args.raw_preds_dir / model_name
        if not model_dir.exists():
            print(f"Warning: {model_dir} not found. Skipping.")
            continue
            
        print(f"Evaluating FP Curve for {model_name}...")
        results[model_name] = compute_curve(model_dir, args.labels_dir, nonempty_stems, args.dist_bin_width, args.max_dist, args.min_preds_per_bin)

    (args.output_dir / "border_fp_curve.json").write_text(json.dumps(results, indent=4), encoding="utf-8")
    plot_curves(results, args.output_dir)
    print(f"\nFigure 4 and JSON data saved to {args.output_dir}")

if __name__ == "__main__":
    main()