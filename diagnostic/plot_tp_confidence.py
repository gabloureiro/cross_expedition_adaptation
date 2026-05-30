#!/usr/bin/env python3
"""
plot_tp_confidence.py — Phase 2: Compute True Positive rates and plot Figure 5.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    plt.style.use("seaborn-v0_8-whitegrid")
except OSError:
    try:
        plt.style.use("seaborn-whitegrid")
    except OSError:
        pass

TILE_SIZE = 640

# ── Geometry helpers ──────────────────────────────────────────────────────────
def yolo_norm_to_xyxy(xc: float, yc: float, w: float, h: float, s: int = TILE_SIZE) -> List[float]:
    return [(xc - w / 2) * s, (yc - h / 2) * s, (xc + w / 2) * s, (yc + h / 2) * s]

def xywh_to_xyxy(xc: float, yc: float, w: float, h: float) -> List[float]:
    return [xc - w / 2, yc - h / 2, xc + w / 2, yc + h / 2]

def iou(a: List[float], b: List[float]) -> float:
    ix1 = max(a[0], b[0]); iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2]); iy2 = min(a[3], b[3])
    iw = max(0.0, ix2 - ix1); ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter == 0: return 0.0
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (area_a + area_b - inter + 1e-9)

def border_dist(box: List[float], tile_size: int = TILE_SIZE) -> float:
    return min(box[0], box[1], tile_size - box[2], tile_size - box[3])

# ── CLI ───────────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate IoU matches and plot TP curve.")
    p.add_argument("--raw_preds_dir",   required=True,  type=Path, help="Path to raw_preds/ containing model folders")
    p.add_argument("--labels_dir",      required=True,  type=Path)
    p.add_argument("--output_dir",      required=True,  type=Path)
    p.add_argument("--models",          nargs="+", default=["YOLOv8", "DINO", "FasterRCNN"])
    p.add_argument("--conf_thresholds", nargs="+", type=float, default=[0.1, 0.3, 0.5, 0.7, 0.8, 0.9])
    p.add_argument("--iou_threshold",   type=float, default=0.5)
    p.add_argument("--border_dist_px",  type=int, default=64)
    return p.parse_args()

# ── Evaluation ────────────────────────────────────────────────────────────────
def compute_tp_rate(raw_dir: Path, labels_dir: Path, conf_thresh: float, iou_thresh: float, border_dist_px: int, nonempty_stems: set) -> float:
    tp, fp = 0, 0
    for jf in raw_dir.glob("*.json"):
        if jf.stem not in nonempty_stems: continue
        gt_path = labels_dir / f"{jf.stem}.txt"
        if not gt_path.exists(): continue
            
        try:
            data = json.loads(jf.read_text(encoding="utf-8"))
            preds = data if isinstance(data, list) else data.get("boxes", [])
        except Exception:
            continue

        gt_boxes = [yolo_norm_to_xyxy(*list(map(float, line.split()))[1:5]) 
                    for line in gt_path.read_text().splitlines() if line.strip()]

        for pred in preds:
            if pred["score"] < conf_thresh: continue
            pred_box = xywh_to_xyxy(*pred["bbox"])
            if border_dist(pred_box, TILE_SIZE) < border_dist_px: continue
            
            if any(iou(pred_box, gt) >= iou_thresh for gt in gt_boxes): tp += 1
            else: fp += 1

    total = tp + fp
    return (tp / total * 100.0) if total > 0 else 0.0

# ── Plotting ──────────────────────────────────────────────────────────────────
def plot_curves(results: Dict[str, List[float]], thresholds: List[float], output_dir: Path):
    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=150)
    styles = {
        "YOLOv8":     {"color": "#e74c3c", "ls": "-",  "marker": "o"},
        "DINO":       {"color": "#2980b9", "ls": "--", "marker": "s"},
        "FasterRCNN": {"color": "#27ae60", "ls": "-.", "marker": "^"}
    }
    default_colors = ["#8e44ad", "#f39c12", "#16a085"]

    for i, (model_name, tp_rates) in enumerate(results.items()):
        s = styles.get(model_name, {"color": default_colors[i % 3], "ls": "-", "marker": "d"})
        ax.plot(thresholds, tp_rates, color=s["color"], linestyle=s["ls"], linewidth=2, 
                marker=s["marker"], markersize=5, label=model_name, zorder=3)

    ax.axvline(0.9, color="black", linestyle=":", linewidth=1.5, alpha=0.7, zorder=4)
    ax.text(0.905, ax.get_ylim()[1] - 5, "operating\npoint", fontsize=8, ha="left", va="top", color="black", zorder=5)
    ax.axhline(50, color="grey", linestyle=":", linewidth=1.0, alpha=0.4, label="_nolegend_", zorder=2)

    ax.set_xlabel("Confidence threshold", fontsize=11)
    ax.set_ylabel("True Positive Rate (%)", fontsize=11)
    ax.set_xlim(0.05, 0.95)
    ax.set_ylim(50, max(90, max([max(r) for r in results.values()] if results else [90]) + 5))
    ax.set_xticks(thresholds)

    ax.grid(True, alpha=0.3, linestyle="--")
    ax.legend(loc="upper left", framealpha=0.9, fontsize=10)

    plt.tight_layout()
    fig.savefig(output_dir / "figure5_tp_confidence.pdf", dpi=300, bbox_inches="tight")
    fig.savefig(output_dir / "figure5_tp_confidence.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    nonempty_stems = {f.stem for f in args.labels_dir.glob("*.txt") if f.read_text(encoding="utf-8").strip()}
    
    results = {}
    for model_name in args.models:
        model_dir = args.raw_preds_dir / model_name
        if not model_dir.exists():
            print(f"Warning: {model_dir} not found. Skipping {model_name}.")
            continue
            
        print(f"\nEvaluating {model_name}...")
        tp_rates = []
        for conf in args.conf_thresholds:
            rate = compute_tp_rate(model_dir, args.labels_dir, conf, args.iou_threshold, args.border_dist_px, nonempty_stems)
            tp_rates.append(rate)
            print(f"  Conf >= {conf:.1f} | TP Rate: {rate:.1f}%")
        results[model_name] = tp_rates

    (args.output_dir / "tp_confidence_data.json").write_text(json.dumps(results, indent=4), encoding="utf-8")
    
    if results:
        plot_curves(results, args.conf_thresholds, args.output_dir)
        print(f"\nPlots and JSON data saved to {args.output_dir}")

if __name__ == "__main__":
    main()