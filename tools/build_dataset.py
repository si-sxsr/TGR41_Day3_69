"""Build the 7-class YOLO-seg dataset (crops around the beans) + full-size GT PNGs for scoring.

Sources
  A. Day 3   data/labels.json + dataset_original/Day3/*.jpg   (hand polygons, 7 classes)
  B. Day 1-2 single-class images + auto-label instance PNGs from the Day 2 repo (class = folder)
  C. Day 2   Roboflow pile photos (good/shell, touching beans)  -> remapped to the 7-class ids
Class ids follow labels.json: 0 husk 1 parchment 2 immature 3 good 4 shell 5 black 6 broken

Usage: python tools/build_dataset.py [--val Mixed_grid6 Mixed_square5 ...] [--all-train]
"""
import argparse
import glob
import json
import os
import shutil

import cv2
import numpy as np

NAMES = ["husk", "parchment", "immature", "good", "shell", "black", "broken"]
CID = {n: i for i, n in enumerate(NAMES)}
D2 = r"C:\Users\ACER\OneDrive\Desktop\TGR-41_day2"
OUT = "yolo7"
MAX_LONG = 1280
DEFAULT_VAL = ["Mixed_grid6", "Mixed_square5", "Mixed_triangle4", "Mixed_square2"]


def crop_box(polys, h, w, margin=0.12):
    P = np.concatenate(polys)
    x0, y0 = P.min(0)
    x1, y1 = P.max(0)
    m = margin * max(x1 - x0, y1 - y0) + 60
    return (int(max(x0 - m, 0)), int(max(y0 - m, 0)), int(min(x1 + m, w)), int(min(y1 + m, h)))


def write_sample(split, name, img, polys, cls, box=None):
    h, w = img.shape[:2]
    if box is None:
        box = crop_box(polys, h, w)
    x0, y0, x1, y1 = box
    crop = img[y0:y1, x0:x1]
    s = min(1.0, MAX_LONG / max(crop.shape[:2]))
    if s < 1:
        crop = cv2.resize(crop, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    ch, cw = crop.shape[:2]
    lines = []
    for p, c in zip(polys, cls):
        q = (np.asarray(p, np.float64) - [x0, y0]) * s
        q = q[:: max(1, len(q) // 200)]  # <=~200 pts is plenty for training
        q[:, 0] = np.clip(q[:, 0] / cw, 0, 1)
        q[:, 1] = np.clip(q[:, 1] / ch, 0, 1)
        lines.append(f"{c} " + " ".join(f"{v:.5f}" for v in q.ravel()))
    cv2.imwrite(f"{OUT}/images/{split}/{name}.jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 95])
    with open(f"{OUT}/labels/{split}/{name}.txt", "w") as f:
        f.write("\n".join(lines) + "\n")


def gt_png(path_png, hw, polys):
    m = np.zeros(hw, np.uint16)
    for i, p in enumerate(polys, 1):
        cv2.fillPoly(m, [np.asarray(p, np.int32)], i)
    cv2.imwrite(path_png, m)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val", nargs="*", default=DEFAULT_VAL)
    ap.add_argument("--all-train", action="store_true", help="put everything in train (final model)")
    a = ap.parse_args()
    val = set() if a.all_train else set(a.val)
    shutil.rmtree(OUT, ignore_errors=True)
    for s in ("train", "val"):
        os.makedirs(f"{OUT}/images/{s}", exist_ok=True)
        os.makedirs(f"{OUT}/labels/{s}", exist_ok=True)
    os.makedirs("val_day3/images", exist_ok=True)
    os.makedirs("val_day3/gt", exist_ok=True)

    # A. Day 3
    for e in json.load(open("data/labels.json", encoding="utf-8")):
        stem = os.path.splitext(e["image_name"])[0]
        img = cv2.imread(f"dataset_original/Day3/{e['image_name']}")
        polys = [np.asarray(o["polygon"]) for o in e["objects"]]
        cls = [CID[o["class"]] for o in e["objects"]]
        split = "val" if stem in val else "train"
        write_sample(split, f"d3_{stem}", img, polys, cls)
        if split == "val":
            shutil.copy(f"dataset_original/Day3/{e['image_name']}", "val_day3/images/")
            gt_png(f"val_day3/gt/{stem}.png", img.shape[:2], polys)
            json.dump([o["class"] for o in e["objects"]], open(f"val_day3/gt/{stem}.json", "w"))

    # B. Day 1-2 single-class images with auto-label instance masks
    for mp in sorted(glob.glob(os.path.join(D2, "labels", "*", "*.png"))):
        c = os.path.basename(os.path.dirname(mp))
        if c not in CID:
            continue
        stem = os.path.splitext(os.path.basename(mp))[0]
        ip = os.path.join(D2, "dataset", c, stem + ".jpg")
        if not os.path.exists(ip):
            continue
        img, m = cv2.imread(ip), cv2.imread(mp, cv2.IMREAD_UNCHANGED)
        if img.shape[:2] != m.shape:
            print("skip size mismatch", ip)
            continue
        polys = []
        for i in range(1, int(m.max()) + 1):
            cnts, _ = cv2.findContours((m == i).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            if cnts:
                cc = max(cnts, key=cv2.contourArea)
                if cv2.contourArea(cc) > 500:
                    polys.append(cc.reshape(-1, 2))
        write_sample("train", f"d12_{c}_{stem}", img, polys, [CID[c]] * len(polys))

    # C. Roboflow piles (0 good, 1 shell) -> 7-class ids; keep image as is
    remap = {0: CID["good"], 1: CID["shell"]}
    for sub, split in (("train", "train"), ("valid", "train" if a.all_train else "val")):
        for ip in sorted(glob.glob(os.path.join(D2, "dataset_rf", sub, "images", "*"))):
            stem = os.path.splitext(os.path.basename(ip))[0]
            lp = os.path.join(D2, "dataset_rf", sub, "labels", stem + ".txt")
            out = []
            for line in open(lp).read().split("\n"):
                if line.strip():
                    v = line.split()
                    out.append(" ".join([str(remap[int(v[0])])] + v[1:]))
            shutil.copy(ip, f"{OUT}/images/{split}/rf_{stem}.jpg")
            open(f"{OUT}/labels/{split}/rf_{stem}.txt", "w").write("\n".join(out) + "\n")

    with open(f"{OUT}/beans7.yaml", "w") as f:
        f.write(f"path: {os.path.abspath(OUT)}\ntrain: images/train\nval: images/val\nnames:\n")
        f.write("".join(f"  {i}: {n}\n" for i, n in enumerate(NAMES)))
    for s in ("train", "val"):
        print(s, len(os.listdir(f"{OUT}/images/{s}")))


if __name__ == "__main__":
    main()
