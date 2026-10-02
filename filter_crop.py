"""Loc va crop anh hoa theo yeu cau 1 cua README.

Moi anh:
  1. Tach hoa bang GrabCut (khoi tao bang hinh chu nhat o giua anh).
  2. Do cac tieu chi: 1 bong duy nhat, nam giua, ti le khung hinh ~ tron
     (xap xi chup thang tu tren xuong), chiem ti le dien tich hop ly.
  3. Dat -> cat vuong quanh tam hoa voi ti le co dinh, resize ve SIZE x SIZE.
Ket qua: dataset/<nguon>/<loai>/*.jpg, manifest.csv (kem so do), rejected.csv.
"""
import argparse
import csv
import glob
import os
import random

import cv2
import numpy as np

SIZE = 256            # kich thuoc anh dau ra
WORK = 160            # canh dai khi phan tich
PAD = 1.30            # canh crop = PAD * canh dai bbox hoa -> hoa luon chiem ti le tuong dong
MIN_AREA, MAX_AREA = 0.12, 0.75   # ti le dien tich hoa / anh (trong khung crop thu duoc)
MAX_CENTER_OFF = 0.18  # lech tam hoa so voi tam anh (theo ti le canh ngan)
MIN_MAIN_RATIO = 0.80  # thanh phan lon nhat / tong vung hoa (1 bong duy nhat)
ASPECT = (0.70, 1.45)  # ti le w/h bbox hoa (gan tron = nhin thang tu tren xuong)
MIN_FILL = 0.55        # dien tich mask / dien tich hinh elip ngoai tiep (loai hoa dai, canh nghieng)
MAX_OVER = 0.12        # cho phep crop lan ra ngoai anh toi da 12% canh (bu vien phan xa)
MAX_BORDER = 0.05      # ti le vien khung crop bi vung "hoa" chiem -> co hoa/vat khac sat canh
MIN_SIDE = 200         # anh goc qua nho thi bo


def segment(img):
    h, w = img.shape[:2]
    s = WORK / max(h, w)
    small = cv2.resize(img, (max(1, int(w * s)), max(1, int(h * s))), interpolation=cv2.INTER_AREA)
    sh, sw = small.shape[:2]
    mask = np.zeros((sh, sw), np.uint8)
    rect = (int(sw * 0.1), int(sh * 0.1), int(sw * 0.8), int(sh * 0.8))
    bgd, fgd = np.zeros((1, 65)), np.zeros((1, 65))
    try:
        cv2.grabCut(small, mask, rect, bgd, fgd, 3, cv2.GC_INIT_WITH_RECT)
    except cv2.error:
        return None, s
    fg = np.where((mask == 1) | (mask == 3), 255, 0).astype(np.uint8)
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    return fg, s


def analyse(img):
    """Tra ve (metrics, crop_box_goc) hoac (metrics, None) neu loai; metrics co 'reason'."""
    h, w = img.shape[:2]
    if min(h, w) < MIN_SIDE:
        return {"reason": "anh_nho"}, None
    fg, s = segment(img)
    if fg is None:
        return {"reason": "grabcut_loi"}, None
    n, lab, stats, cents = cv2.connectedComponentsWithStats(fg)
    if n <= 1:
        return {"reason": "khong_thay_hoa"}, None
    areas = stats[1:, cv2.CC_STAT_AREA]
    k = 1 + int(np.argmax(areas))
    total = float(areas[areas > 0.01 * fg.size].sum()) or 1.0
    main_ratio = stats[k, cv2.CC_STAT_AREA] / total
    x, y, bw, bh, a = stats[k]
    sh, sw = fg.shape
    cx, cy = cents[k]
    aspect = bw / bh
    fill = a / (np.pi / 4 * bw * bh)
    side = PAD * max(bw, bh)
    # canh crop (he toa do anh phan tich), co the vuot bien -> se pad thanh dai mau phan xa
    area_ratio = a / (side * side)
    off = float(np.hypot(cx - sw / 2, cy - sh / 2)) / min(sw, sh)
    m = dict(main_ratio=round(main_ratio, 3), aspect=round(aspect, 3), fill=round(fill, 3),
             area_ratio=round(area_ratio, 3), off=round(off, 3), reason="")
    # tam crop = tam hoa, crop phai nam trong anh goc
    x0, y0 = (cx - side / 2) / s, (cy - side / 2) / s
    side_o = side / s
    over = max(-x0, -y0, x0 + side_o - w, y0 + side_o - h, 0) / side_o
    m["over"] = round(float(over), 3)
    # vung hoa chiem bao nhieu % vien khung crop (phat hien cum hoa / vat the khac sat canh)
    R = int(round(side))
    X0, Y0 = int(round(cx - side / 2)), int(round(cy - side / 2))
    pad = cv2.copyMakeBorder(fg, R, R, R, R, cv2.BORDER_CONSTANT, value=0)
    crop = pad[Y0 + R:Y0 + R + R, X0 + R:X0 + R + R] > 0
    t = max(2, R // 16)
    ring = np.ones_like(crop); ring[t:-t, t:-t] = False
    border = float(crop[ring].mean()) if ring.any() else 0.0
    m["border"] = round(border, 3)
    if over > MAX_OVER:
        m["reason"] = "hoa_sat_bien_khong_crop_duoc"
    elif border > MAX_BORDER:
        m["reason"] = "nhieu_hoa_hoac_nen_roi"
    elif main_ratio < MIN_MAIN_RATIO:
        m["reason"] = "nhieu_hoa_hoac_nen_roi"
    elif not (ASPECT[0] <= aspect <= ASPECT[1]):
        m["reason"] = "khong_phai_goc_nhin_tren_xuong(dai)"
    elif fill < MIN_FILL:
        m["reason"] = "hinh_dang_khong_tron"
    elif not (MIN_AREA <= area_ratio <= MAX_AREA):
        m["reason"] = "hoa_qua_nho_hoac_qua_lon"
    elif off > MAX_CENTER_OFF:
        m["reason"] = "lech_tam"
    if m["reason"]:
        return m, None
    return m, (int(round(x0)), int(round(y0)), int(round(side_o)))


def dhash(img, n=8):
    g = cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (n + 1, n), interpolation=cv2.INTER_AREA)
    return int("".join("1" if v else "0" for v in (g[:, 1:] > g[:, :-1]).flatten()), 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="dataset")
    ap.add_argument("--sample", type=int, default=0, help="chi chay thu tren N anh ngau nhien")
    ap.add_argument("--jobs", type=int, default=4)
    args = ap.parse_args()

    items = []
    for f in sorted(glob.glob("flowers/*/*")):
        items.append((f, "flowers", f.split(os.sep)[1]))
    for f in sorted(glob.glob("jpg/*")):
        items.append((f, "jpg", "unlabeled"))
    if args.sample:
        random.seed(0)
        items = random.sample(items, args.sample)

    from multiprocessing import Pool
    with Pool(args.jobs) as p:
        results = p.map(process, items, chunksize=16)

    os.makedirs(args.out, exist_ok=True)
    seen, kept, rej = [], [], []
    for (path, src, label), (m, crop) in zip(items, results):
        if crop is None:
            rej.append(dict(path=path, **m))
            continue
        img = cv2.imread(path)
        x, y, sd = crop
        p = sd
        img = cv2.copyMakeBorder(img, p, p, p, p, cv2.BORDER_REFLECT)
        out = cv2.resize(img[y + p:y + p + sd, x + p:x + p + sd], (SIZE, SIZE), interpolation=cv2.INTER_AREA)
        hs = dhash(out)
        if any(bin(hs ^ o).count("1") <= 4 for o in seen):  # trung/gan trung anh da co
            rej.append(dict(path=path, reason="trung_lap", **{k: v for k, v in m.items() if k != "reason"}))
            continue
        seen.append(hs)
        d = os.path.join(args.out, src, label)
        os.makedirs(d, exist_ok=True)
        dst = os.path.join(d, os.path.splitext(os.path.basename(path))[0] + ".jpg")
        cv2.imwrite(dst, out, [cv2.IMWRITE_JPEG_QUALITY, 95])
        kept.append(dict(file=dst, source=path, label=label, **{k: v for k, v in m.items() if k != "reason"}))

    keys = ["file", "source", "label", "main_ratio", "aspect", "fill", "area_ratio", "off", "over", "border"]
    with open(os.path.join(args.out, "manifest.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, keys); w.writeheader(); w.writerows(kept)
    with open(os.path.join(args.out, "rejected.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, ["path", "reason", "main_ratio", "aspect", "fill", "area_ratio", "off", "over", "border"], extrasaction="ignore")
        w.writeheader(); w.writerows(rej)
    from collections import Counter
    print(f"Tong {len(items)} | giu {len(kept)} | loai {len(rej)}")
    print("Ly do loai:", Counter(r["reason"] for r in rej).most_common())
    print("Giu theo nhan:", Counter(k["label"] for k in kept).most_common())


def process(item):
    img = cv2.imread(item[0])
    if img is None:
        return {"reason": "khong_doc_duoc"}, None
    return analyse(img)


if __name__ == "__main__":
    main()
