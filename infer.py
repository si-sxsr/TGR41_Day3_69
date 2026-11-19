"""Day 3 inference: instance segmentation + 7-class classification of coffee beans (Raspberry Pi CPU).

python infer.py --input test/ --weights weights/ --out outputs/

Outputs
  outputs/seg_mask/<stem>.png    single-channel ids, 0 = background, 1..N = bean (full size after EXIF)
  outputs/predictions.json       [{"image_name", "objects": [{"id", "class", "polygon"}]}]   (= results.json for HTTP)
  outputs/mqtt/<stem>.json       {"image", "latency_ms", "counts"(7 keys), "defect_ratio"}  (what is published)
  outputs/vis/<stem>.jpg         overlay for eyes only

Pipeline per image: reduced JPEG decode -> find bean region on a 256 px thumbnail (white paper) ->
crop -> YOLO-seg (NCNN) -> one pixel one bean -> per-bean upsample to full size -> PNG/polygons/JSON.
CPU only, no GUI, no downloads, no TTA.
"""
import argparse
import glob
import json
import os
import sys
import time

import cv2
import numpy as np

CLASSES = ["good", "shell", "black", "husk", "broken", "parchment", "immature"]  # counts key order (slide 7)
DEFECT = [c for c in CLASSES if c != "good"]
EXTS = (".jpg", ".jpeg", ".png")
WORK_LONG = 1000  # decode at the smallest JPEG scale (1/2, 1/4) whose long side is still >= this


def die(msg):
    print(f"ERROR: {msg}", file=sys.stderr)
    sys.exit(1)


def find_weights(folder):
    """Candidates in priority order: *_ncnn_model/ > *.onnx > *.pt."""
    if not os.path.isdir(folder):
        die(f"weights folder not found: {folder}")
    cands = []
    for pat in ("*_ncnn_model", "*.onnx", "*.pt"):
        cands += sorted(glob.glob(os.path.join(folder, pat)))
    if not cands:
        die(f"no *_ncnn_model / *.onnx / *.pt in {folder}")
    return cands


def load_model(folder, imgsz):
    from ultralytics import YOLO
    for w in find_weights(folder):
        try:
            m = YOLO(w, task="segment")
            for _ in range(2):  # warm-up
                m.predict(np.full((imgsz, imgsz, 3), 200, np.uint8), imgsz=imgsz, verbose=False, device="cpu")
            return m, w
        except Exception as e:
            print(f"WARNING: cannot load {w}: {e}", file=sys.stderr)
    die("no loadable model in weights folder")


def full_size(path, data=None):
    """Image size after EXIF orientation, read from the header only (file path or raw bytes)."""
    import io
    from PIL import Image
    with Image.open(io.BytesIO(data) if data is not None else path) as im:
        w, h = im.size
        if im.getexif().get(274, 1) in (5, 6, 7, 8):
            w, h = h, w
    return h, w


def read_work(path, h, w, data=None):
    """Decode at reduced size (fast on Pi); cv2 applies EXIF orientation. data = raw bytes (no disk)."""
    flag = cv2.IMREAD_COLOR
    for f, r in ((cv2.IMREAD_REDUCED_COLOR_4, 4), (cv2.IMREAD_REDUCED_COLOR_2, 2)):
        if max(h, w) / r >= WORK_LONG:
            flag = f
            break
    if data is not None:
        img = cv2.imdecode(np.frombuffer(data, np.uint8), flag)
    else:
        img = cv2.imread(path, flag)
    if img is None:
        die(f"cannot read image: {path}")
    return img


def bean_roi(img, margin=0.12, pad_full=60, full_long=4096):
    """Bounding box (work-image coords) of everything darker/more coloured than the white paper.
    Same margin rule as training crops. Returns None -> use whole image."""
    h, w = img.shape[:2]
    s = 256 / max(h, w)
    sm = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    lab = cv2.cvtColor(sm, cv2.COLOR_BGR2LAB).astype(np.float32)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))
    bg_l = cv2.GaussianBlur(cv2.morphologyEx(lab[..., 0], cv2.MORPH_CLOSE, k), (0, 0), 6)
    med_a, med_b = np.median(lab[..., 1]), np.median(lab[..., 2])
    score = np.clip(bg_l - lab[..., 0], 0, None) + 1.5 * np.hypot(lab[..., 1] - med_a, lab[..., 2] - med_b)
    fg = (score > 18).astype(np.uint8)
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    n, lab_cc, st, _ = cv2.connectedComponentsWithStats(fg)
    hs, ws = fg.shape
    keep = []
    for i in range(1, n):
        x, y, bw, bh, a = st[i]
        if a < 6 or a > 0.25 * hs * ws:
            continue
        if x <= 0 or y <= 0 or x + bw >= ws or y + bh >= hs:  # paper edge / table / shadow at border
            continue
        keep.append((x, y, x + bw, y + bh))
    if not keep:
        return None
    k = np.array(keep, np.float32) / s
    x0, y0 = k[:, 0].min(), k[:, 1].min()
    x1, y1 = k[:, 2].max(), k[:, 3].max()
    m = margin * max(x1 - x0, y1 - y0) + pad_full * max(h, w) / full_long
    x0, y0 = int(max(x0 - m, 0)), int(max(y0 - m, 0))
    x1, y1 = int(min(x1 + m, w)), int(min(y1 + m, h))
    if (x1 - x0) * (y1 - y0) > 0.8 * h * w or x1 - x0 < 16 or y1 - y0 < 16:
        return None
    return x0, y0, x1, y1


def predict_instances(model, img, imgsz, conf):
    """List of (cls_id, score, bool mask at mask resolution covering img) sorted by score desc."""
    r = model.predict(img, imgsz=imgsz, conf=conf, verbose=False, device="cpu")[0]
    if r.masks is None or len(r.boxes) == 0:
        return []
    masks = r.masks.data.cpu().numpy() > 0.5
    h0, w0 = img.shape[:2]
    mh, mw = masks.shape[1:]
    if (mh, mw) != (h0, w0):  # remove letterbox padding
        g = min(mh / h0, mw / w0)
        px, py = (mw - w0 * g) / 2, (mh - h0 * g) / 2
        y0, x0 = int(round(py - 0.1)), int(round(px - 0.1))
        y1, x1 = int(round(mh - py + 0.1)), int(round(mw - px + 0.1))
        masks = masks[:, y0:y1, x0:x1]
    cls = r.boxes.cls.cpu().numpy().astype(int)
    sc = r.boxes.conf.cpu().numpy()
    order = np.argsort(-sc)
    return [(cls[i], float(sc[i]), masks[i]) for i in order]


def build_instance_map(insts, min_frac=2e-4, min_keep=0.5):
    """Paint highest score first; each pixel belongs to one bean. A bean that lost most of its
    area to higher-scoring beans is a duplicate -> dropped."""
    if not insts:
        return np.zeros((1, 1), np.uint16), []
    hw, ww = insts[0][2].shape
    small = np.zeros((hw, ww), np.uint16)
    min_area = max(3, int(min_frac * hw * ww))
    kept = []
    for c, s, m in insts:
        area = int(m.sum())
        free = m & (small == 0)
        nf = int(free.sum())
        if nf < min_area or nf < min_keep * area:
            continue
        kept.append((c, s))
        small[free] = len(kept)
    return small, kept


def finalize(small, kept, names, full_hw, box_full, eps=1.0):
    """Upscale each instance on its own crop (bilinear), keep largest part, fill holes, paint into the
    full-size map so that mask == polygon interior. box_full = (X0, Y0, X1, Y1) region `small` covers."""
    from scipy.ndimage import find_objects
    H, W = full_hw
    X0, Y0, X1, Y1 = box_full
    hs, ws = small.shape
    sy, sx = (Y1 - Y0) / hs, (X1 - X0) / ws
    out = np.zeros((H, W), np.uint16)
    objs = []
    n = 0
    for idx, sl in enumerate(find_objects(small)):
        if sl is None:
            continue
        i = idx + 1
        ys0, ys1 = max(sl[0].start - 2, 0), min(sl[0].stop + 2, hs)
        xs0, xs1 = max(sl[1].start - 2, 0), min(sl[1].stop + 2, ws)
        y0, y1 = Y0 + int(round(ys0 * sy)), min(Y0 + int(round(ys1 * sy)), H)
        x0, x1 = X0 + int(round(xs0 * sx)), min(X0 + int(round(xs1 * sx)), W)
        if y1 - y0 < 2 or x1 - x0 < 2:
            continue
        m = (small[ys0:ys1, xs0:xs1] == i).astype(np.float32)
        m = cv2.resize(m, (x1 - x0, y1 - y0), interpolation=cv2.INTER_LINEAR) > 0.5
        region = out[y0:y1, x0:x1]
        m &= region == 0
        cnts, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not cnts:
            continue
        c = max(cnts, key=cv2.contourArea)
        if eps > 0:  # fewer points -> smaller JSON / faster POST; mask is drawn from the same polygon
            c = cv2.approxPolyDP(c, eps, True)
        if len(c) < 3 or cv2.contourArea(c) < 4:
            continue
        keep = np.zeros(m.shape, np.uint8)
        cv2.drawContours(keep, [c], -1, 1, thickness=cv2.FILLED)
        keep = (keep > 0) & (region == 0)
        n += 1
        region[keep] = n
        pts = c.reshape(-1, 2) + [x0, y0]
        objs.append({"id": n, "class": str(names[int(kept[i - 1][0])]).lower(), "polygon": pts.astype(int).tolist()})
    return out, objs


def mqtt_payload(image_name, objs, latency_ms):
    counts = {c: 0 for c in CLASSES}
    for o in objs:
        counts[o["class"]] += 1
    n = len(objs)
    return {"image": image_name, "latency_ms": int(round(latency_ms)), "counts": counts,
            "defect_ratio": round(sum(counts[c] for c in DEFECT) / n, 4) if n else 0.0}


def draw_vis(img_small, full_hw, objs, path):
    h, w = img_small.shape[:2]
    k = w / full_hw[1]
    s = min(1.0, 1200 / max(h, w))
    if s < 1:
        img_small = cv2.resize(img_small, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        k *= s
    vis = img_small.copy()
    over = vis.copy()
    col = {c: tuple(int(v) for v in np.random.default_rng(i + 3).integers(40, 255, 3)) for i, c in enumerate(CLASSES)}
    for o in objs:
        cv2.fillPoly(over, [(np.asarray(o["polygon"]) * k).astype(np.int32)], col.get(o["class"], (0, 255, 0)))
    vis = cv2.addWeighted(over, 0.45, vis, 0.55, 0)
    for o in objs:
        pts = (np.asarray(o["polygon"]) * k).astype(np.int32)
        cv2.polylines(vis, [pts], True, (255, 255, 255), 1)
        x, y = pts.min(0)
        cv2.putText(vis, f'{o["id"]}:{o["class"]}', (int(x), max(int(y) - 3, 10)), 0, 0.45, (0, 0, 255), 1)
    cv2.imwrite(path, vis, [cv2.IMWRITE_JPEG_QUALITY, 85])


def list_images(folder):
    if not os.path.isdir(folder):
        die(f"input folder not found: {folder}")
    files = sorted(f for f in os.listdir(folder) if f.lower().endswith(EXTS) and os.path.isfile(os.path.join(folder, f)))
    if not files:
        die(f"no .jpg/.jpeg/.png images in {folder}")
    stems = [os.path.splitext(f)[0].lower() for f in files]
    if len(set(stems)) != len(stems):
        die("duplicate image stems in input (PNG outputs would overwrite each other)")
    return files


class Inferencer:
    def __init__(self, weights, imgsz=640, conf=0.15, crop=True, vis=True, eps=1.0):
        self.eps = eps
        self.model, self.wpath = load_model(weights, imgsz)
        self.names = self.model.names
        bad = [n for n in self.names.values() if str(n).lower() not in CLASSES]
        if bad:
            die(f"model classes not in spec: {bad}")
        self.imgsz, self.conf, self.crop, self.vis = imgsz, conf, crop, vis

    def run_one(self, path, out):
        """Process one image; writes seg_mask/vis/mqtt files. Returns (prediction entry, mqtt payload)."""
        res = self.predict(path)
        self.save(res, out)
        return res["entry"], res["mqtt"]

    def predict(self, path, name=None, t0=None, data=None):
        """Compute only (no file writes) -> dict with entry/mqtt payload + data for save().
        data = raw image bytes (skips the disk). Latency counts from t0 (default: now)."""
        t0 = time.perf_counter() if t0 is None else t0
        f = name or os.path.basename(path)
        try:
            H, W = full_size(path, data)
        except Exception as e:
            die(f"cannot read image {path}: {e}")
        img = read_work(path, H, W, data)
        hw, ww = img.shape[:2]
        box = bean_roi(img, full_long=max(H, W)) if self.crop else None
        if box is None:
            box = (0, 0, ww, hw)
        x0, y0, x1, y1 = box
        insts = predict_instances(self.model, img[y0:y1, x0:x1], self.imgsz, self.conf)
        small, kept = build_instance_map(insts)
        fx, fy = W / ww, H / hw
        box_full = (int(round(x0 * fx)), int(round(y0 * fy)), min(int(round(x1 * fx)), W), min(int(round(y1 * fy)), H))
        full, objs = finalize(small, kept, self.names, (H, W), box_full, self.eps)
        latency = (time.perf_counter() - t0) * 1000
        return {"entry": {"image_name": f, "objects": objs}, "mqtt": mqtt_payload(f, objs, latency),
                "full": full, "img": img, "hw": (H, W)}

    def save(self, res, out):
        """Write seg_mask PNG, mqtt JSON and vis for a predict() result."""
        f = res["entry"]["image_name"]
        stem = os.path.splitext(f)[0]
        full = res["full"]
        if full.max() <= 255:
            full = full.astype(np.uint8)
        if not cv2.imwrite(os.path.join(out, "seg_mask", stem + ".png"), full,
                           [cv2.IMWRITE_PNG_COMPRESSION, 1, cv2.IMWRITE_PNG_STRATEGY, cv2.IMWRITE_PNG_STRATEGY_RLE]):
            die(f"failed to write mask for {f}")
        with open(os.path.join(out, "mqtt", stem + ".json"), "w", encoding="utf-8") as fh:
            json.dump(res["mqtt"], fh, ensure_ascii=False)
        if self.vis:
            draw_vis(res["img"], res["hw"], res["entry"]["objects"], os.path.join(out, "vis", stem + ".jpg"))


def prepare_out(out, vis=True):
    for d in ("seg_mask", "mqtt") + (("vis",) if vis else ()):
        os.makedirs(os.path.join(out, d), exist_ok=True)


def write_predictions(out, results):
    with open(os.path.join(out, "predictions.json"), "w", encoding="utf-8") as fh:
        json.dump(results, fh, ensure_ascii=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.15)
    ap.add_argument("--no-crop", action="store_true", help="run the model on the whole image")
    ap.add_argument("--no-vis", action="store_true")
    ap.add_argument("--eps", type=float, default=1.0, help="polygon simplification (full-res px), 0 = off")
    ap.add_argument("--work-long", type=int, default=WORK_LONG, help="decode scale: smallest with long side >= this")
    a = ap.parse_args()
    globals()["WORK_LONG"] = a.work_long

    files = list_images(a.input)
    if os.path.abspath(a.out) == os.path.abspath(a.input):
        die("--out must differ from --input")
    inf = Inferencer(a.weights, a.imgsz, a.conf, not a.no_crop, not a.no_vis, a.eps)
    prepare_out(a.out, not a.no_vis)
    results, times = [], []
    for f in files:
        t0 = time.perf_counter()
        r, pay = inf.run_one(os.path.join(a.input, f), a.out)
        results.append(r)
        times.append(time.perf_counter() - t0)
        print(f"{f}: {len(r['objects'])} beans {pay['counts']} | {times[-1] * 1000:.0f} ms")
    write_predictions(a.out, results)
    print(f"done {len(files)} images | median {np.median(times) * 1000:.0f} ms/image | model {inf.wpath}")


if __name__ == "__main__":
    main()
