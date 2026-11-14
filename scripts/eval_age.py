"""Оценка возраста на UTKFace (или любой папке с именами [age]_[gender]_[race]_...jpg).

  python scripts/eval_age.py --data data/UTKFace --list models/test_files.txt
Выводит MAE, MAE по группам (возраст/пол/раса) и точность на порогах 18 и 21.
"""
import argparse, glob, os, sys
from collections import defaultdict

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.age import AgeEstimator          # noqa: E402
from app.config import settings           # noqa: E402

GROUPS = [(0, 12), (13, 17), (18, 25), (26, 40), (41, 60), (61, 120)]
RACE = {0: "White", 1: "Black", 2: "Asian", 3: "Indian", 4: "Other"}
GENDER = {0: "male", 1: "female"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True); ap.add_argument("--list")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    files = open(a.list).read().split() if a.list else sorted(glob.glob(os.path.join(a.data, "*.jpg")))
    if a.limit:
        files = files[:a.limit]
    est = AgeEstimator(settings.age_model, settings.mock)
    rows = []
    for f in files:
        p = os.path.basename(f).split("_")
        try:
            age, g, r = int(p[0]), int(p[1]), int(p[2])
        except (ValueError, IndexError):
            continue
        img = cv2.imread(f if os.path.exists(f) else os.path.join(a.data, f))
        if img is None:
            continue
        rows.append((age, g, r, est.predict(img).age))
    arr = np.array(rows, dtype=float)
    err = np.abs(arr[:, 3] - arr[:, 0])
    print(f"N={len(arr)}  MAE={err.mean():.2f}")
    print("\nMAE по возрастным группам:")
    for lo, hi in GROUPS:
        m = (arr[:, 0] >= lo) & (arr[:, 0] <= hi)
        if m.any(): print(f"  {lo:>3}-{hi:<3} n={m.sum():<5} MAE={err[m].mean():.2f}")
    for name, col, names in (("пол", 1, GENDER), ("раса", 2, RACE)):
        print(f"\nMAE по полю '{name}':")
        for k, v in names.items():
            m = arr[:, col] == k
            if m.any(): print(f"  {v:<7} n={m.sum():<5} MAE={err[m].mean():.2f}")
    print("\nТочность на порогах (pred>=T совпадает с true>=T):")
    for T in (18, 21):
        print(f"  T={T}: {((arr[:, 3] >= T) == (arr[:, 0] >= T)).mean():.3f}")


if __name__ == "__main__":
    main()
