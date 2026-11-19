"""Train the Day 3 7-class bean instance-segmentation model (CPU is fine, ~70 images).

1) python tools/build_dataset.py              # yolo7/ (crops around beans) + val_day3/ GT
   python tools/build_dataset.py --all-train  # final model: everything in train
2) python train.py --model <init.pt> --name run --epochs 80 --imgsz 640
3) export: python train.py --export runs/segment/<name>/weights/best.pt --imgsz 640
"""
import argparse
import time

from ultralytics import YOLO

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="yolo11n-seg.pt", help="init weights (.pt) or arch (.yaml)")
ap.add_argument("--data", default="yolo7/beans7.yaml")
ap.add_argument("--name", default="beans7")
ap.add_argument("--epochs", type=int, default=80)
ap.add_argument("--imgsz", type=int, default=640)
ap.add_argument("--batch", type=int, default=8)
ap.add_argument("--device", default="cpu")
ap.add_argument("--export", default=None, help="export this .pt to ONNX + NCNN and exit")
a = ap.parse_args()

if a.export:
    m = YOLO(a.export)
    m.export(format="onnx", imgsz=a.imgsz, simplify=True)
    YOLO(a.export).export(format="ncnn", imgsz=a.imgsz)
    raise SystemExit

t = time.time()
YOLO(a.model).train(
    data=a.data, imgsz=a.imgsz, epochs=a.epochs, batch=a.batch, device=a.device, workers=2,
    cache="ram", project="runs/segment", name=a.name, exist_ok=True, plots=False,
    mosaic=1.0, close_mosaic=10, copy_paste=0.5, degrees=180, flipud=0.5, fliplr=0.5,
    scale=0.5, hsv_h=0.01, hsv_s=0.4, hsv_v=0.3, patience=200, seed=0, deterministic=True)
print(f"### {a.name} done in {(time.time() - t) / 60:.1f} min", flush=True)
