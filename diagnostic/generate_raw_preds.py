#!/usr/bin/env python3
"""
generate_raw_preds.py — Phase 1: Run teacher models to generate raw predictions.
Saves predictions (conf >= 0.001) to output_dir/raw_preds/{ModelName}/{stem}.json
"""

import argparse
import gc
import json
from pathlib import Path
from typing import List

import torch

# ── Optional model backends ───────────────────────────────────────────────────
MMDET_AVAILABLE = False
try:
    from mmdet.apis import init_detector, inference_detector
    MMDET_AVAILABLE = True
except ImportError:
    pass

YOLO_AVAILABLE = False
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    pass

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

def release_gpu(model) -> None:
    del model
    gc.collect()
    torch.cuda.empty_cache()

# ── CLI ───────────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate raw predictions for Figure 5.")
    p.add_argument("--images_dir",            required=True,  type=Path)
    p.add_argument("--labels_dir",            required=True,  type=Path)
    p.add_argument("--output_dir",            required=True,  type=Path)
    
    p.add_argument("--yolo_checkpoint",       type=Path, default=None)
    p.add_argument("--dino_config",           type=Path, default=None)
    p.add_argument("--dino_checkpoint",       type=Path, default=None)
    p.add_argument("--fasterrcnn_config",     type=Path, default=None)
    p.add_argument("--fasterrcnn_checkpoint", type=Path, default=None)
    
    p.add_argument("--device",               default="cuda:0")
    p.add_argument("--inference_batch_size", type=int, default=16)
    p.add_argument("--conf_inference",       type=float, default=0.001)
    p.add_argument("--resume",               action="store_true")
    return p.parse_args()

# ── Inference ─────────────────────────────────────────────────────────────────
def run_yolov8(images: List[Path], out_dir: Path, checkpoint: Path, conf: float, batch_size: int, device: str, resume: bool):
    if not YOLO_AVAILABLE: return
    todo = [p for p in images if not (out_dir / f"{p.stem}.json").exists()] if resume else images
    if not todo: return
    
    model = YOLO(str(checkpoint))
    paths_str = [str(p) for p in todo]

    for batch_start in range(0, len(paths_str), batch_size):
        batch = paths_str[batch_start: batch_start + batch_size]
        for result in model.predict(source=batch, conf=conf, iou=0.45, imgsz=640, device=device, stream=True, verbose=False, save=False):
            stem, boxes = Path(result.path).stem, []
            if result.boxes is not None and len(result.boxes):
                for xyxy, score, cls in zip(result.boxes.xyxy.cpu().numpy(), result.boxes.conf.cpu().numpy(), result.boxes.cls.cpu().numpy()):
                    if int(cls) == 0:
                        x1, y1, x2, y2 = [float(v) for v in xyxy]
                        w, h = x2 - x1, y2 - y1
                        boxes.append({"bbox": [x1 + w / 2, y1 + h / 2, w, h], "score": float(score)})
            (out_dir / f"{stem}.json").write_text(json.dumps({"boxes": boxes}), encoding="utf-8")
    release_gpu(model)

def run_mmdet(model_name: str, images: List[Path], out_dir: Path, config: Path, checkpoint: Path, conf: float, device: str, resume: bool):
    if not MMDET_AVAILABLE: return
    todo = [p for p in images if not (out_dir / f"{p.stem}.json").exists()] if resume else images
    if not todo: return

    model = init_detector(str(config), str(checkpoint), device=device)
    for img_path in todo:
        try:
            with torch.no_grad():
                result = inference_detector(model, str(img_path))
            boxes_out = []
            if hasattr(result, "pred_instances"):
                instances = result.pred_instances
                for j in range(len(instances.bboxes)):
                    if instances.scores[j] >= conf and int(instances.labels[j]) == 0:
                        x1, y1, x2, y2 = [float(v) for v in instances.bboxes.cpu().numpy()[j]]
                        w, h = x2 - x1, y2 - y1
                        boxes_out.append({"bbox": [x1 + w / 2, y1 + h / 2, w, h], "score": float(instances.scores[j])})
            (out_dir / f"{img_path.stem}.json").write_text(json.dumps({"boxes": boxes_out}), encoding="utf-8")
        except Exception as exc:
            print(f"  [{model_name}] Warning: inference failed on {img_path.name}: {exc}")
    release_gpu(model)

def main():
    args = parse_args()
    raw_preds_dir = args.output_dir / "raw_preds"
    raw_preds_dir.mkdir(parents=True, exist_ok=True)

    specs = []
    if args.yolo_checkpoint: specs.append({"name": "YOLOv8", "type": "yolo", "ckpt": args.yolo_checkpoint})
    if args.dino_config and args.dino_checkpoint: specs.append({"name": "DINO", "type": "mmdet", "config": args.dino_config, "ckpt": args.dino_checkpoint})
    if args.fasterrcnn_config and args.fasterrcnn_checkpoint: specs.append({"name": "FasterRCNN", "type": "mmdet", "config": args.fasterrcnn_config, "ckpt": args.fasterrcnn_checkpoint})

    images = sorted([p for p in args.images_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS])
    labeled = [p for p in images if (args.labels_dir / f"{p.stem}.txt").exists()]
    print(f"Found {len(labeled)} labeled tiles.")

    for spec in specs:
        out_dir = raw_preds_dir / spec["name"]
        out_dir.mkdir(exist_ok=True)
        print(f"Processing {spec['name']}...")
        if spec["type"] == "yolo":
            run_yolov8(labeled, out_dir, spec["ckpt"], args.conf_inference, args.inference_batch_size, args.device, args.resume)
        else:
            run_mmdet(spec["name"], labeled, out_dir, spec["config"], spec["ckpt"], args.conf_inference, args.device, args.resume)
            
    print(f"\nDone. Raw predictions saved to {raw_preds_dir}")

if __name__ == "__main__":
    main()