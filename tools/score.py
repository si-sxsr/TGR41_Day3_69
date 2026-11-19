"""Score outputs/ against GT instance PNGs with the Day 3 rules (slide 12) + class accuracy + MQTT check.

P1 = 15 * mean_img[ sum IoU of 1:1 Hungarian matches / max(#gt, #pred) ]   (class-agnostic)
P2 = 15 * mean_img[ #matched (IoU>0) / max(#gt, #pred) ]
Latency (25): n = ceil(t-1) steps of 5; scaled by P/15 if P < 15.
MQTT (5): counts in mqtt/<stem>.json == class counts of the beans in seg_mask/predictions.json
Usage: python tools/score.py --pred outputs --gt-dir val_day3/gt [--latency 0.8]
"""
import argparse
import glob
import json
import math
import os
from collections import Counter

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment


def iou_matrix(gt, pr):
    g_ids, p_ids = np.unique(gt)[1:], np.unique(pr)[1:]
    if len(g_ids) == 0 or len(p_ids) == 0:
        return np.zeros((len(g_ids), len(p_ids))), g_ids, p_ids
    gmax, pmax = int(gt.max()) + 1, int(pr.max()) + 1
    joint = np.bincount(gt.astype(np.int64).ravel() * pmax + pr.astype(np.int64).ravel(),
                        minlength=gmax * pmax).reshape(gmax, pmax)
    ga, pa = joint.sum(1), joint.sum(0)
    inter = joint[np.ix_(g_ids, p_ids)].astype(float)
    return inter / np.maximum(ga[g_ids][:, None] + pa[p_ids][None, :] - inter, 1), g_ids, p_ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", default="outputs")
    ap.add_argument("--gt-dir", required=True)
    ap.add_argument("--latency", type=float, default=None)
    a = ap.parse_args()
    preds = {d["image_name"]: d for d in json.load(open(os.path.join(a.pred, "predictions.json"), encoding="utf-8"))}
    ious, cnts = [], []
    ok_cls = tot_cls = mqtt_ok = 0
    gpaths = sorted(glob.glob(os.path.join(a.gt_dir, "*.png")))
    conf = Counter()
    for gpath in gpaths:
        stem = os.path.splitext(os.path.basename(gpath))[0]
        gt = cv2.imread(gpath, cv2.IMREAD_UNCHANGED)
        pp = os.path.join(a.pred, "seg_mask", stem + ".png")
        pr = cv2.imread(pp, cv2.IMREAD_UNCHANGED) if os.path.exists(pp) else np.zeros_like(gt)
        assert pr.shape == gt.shape, f"{stem}: shape {pr.shape} vs gt {gt.shape}"
        M, g_ids, p_ids = iou_matrix(gt, pr)
        denom = max(len(g_ids), len(p_ids), 1)
        r, c = linear_sum_assignment(-M) if M.size else (np.array([], int), np.array([], int))
        v = M[r, c]
        ious.append(v.sum() / denom)
        cnts.append((v > 0).sum() / denom)
        entry = next((d for k, d in preds.items() if os.path.splitext(k)[0] == stem), {"objects": []})
        pcls = {o["id"]: o["class"] for o in entry["objects"]}
        gjs = os.path.join(a.gt_dir, stem + ".json")
        if os.path.exists(gjs):
            gcls = json.load(open(gjs))
            for ri, ci in zip(r, c):
                if M[ri, ci] > 0:
                    g, p = gcls[g_ids[ri] - 1], pcls.get(int(p_ids[ci]), "?")
                    conf[(g, p)] += 1
                    ok_cls += g == p
                    tot_cls += 1
        mp = os.path.join(a.pred, "mqtt", stem + ".json")
        if os.path.exists(mp):
            m = json.load(open(mp, encoding="utf-8"))
            seg_ids = set(int(i) for i in np.unique(pr)[1:])
            want = Counter(o["class"] for o in entry["objects"] if o["id"] in seg_ids)
            ok = len(m["counts"]) == 7 and all(m["counts"][k] == want.get(k, 0) for k in m["counts"]) \
                and len(seg_ids) == len(entry["objects"])
            mqtt_ok += ok
        print(f"{stem:24s} gt {len(g_ids):2d} pred {len(p_ids):2d}  mIoU(matched) {v.mean() if len(v) else 0:.3f}"
              f"  IoU-score {ious[-1]:.3f}  count {cnts[-1]:.3f}")
    P1, P2 = 15 * np.mean(ious), 15 * np.mean(cnts)
    print(f"P1 IoU {P1:.2f}/15  P2 count {P2:.2f}/15  P {P1 + P2:.2f}/30  | class acc {ok_cls}/{tot_cls}"
          f"  | MQTT {5 * mqtt_ok / len(gpaths):.2f}/5")
    errs = {k: v for k, v in conf.items() if k[0] != k[1]}
    if errs:
        print("class confusions (gt->pred):", dict(sorted(errs.items(), key=lambda x: -x[1])))
    if a.latency is not None:
        n = max(0, math.ceil(round(a.latency - 1, 6)))
        L = max(0, 25 - 5 * n) * (min(P1 + P2, 15) / 15)
        print(f"latency {a.latency:.2f}s -> {L:.1f}/25   TOTAL = {P1 + P2 + L + 5 * mqtt_ok / len(gpaths):.2f}/60")


if __name__ == "__main__":
    main()
