"""Замер latency пайплайна: python scripts/benchmark.py --image face.jpg --n 100"""
import argparse, os, sys, time

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.config import settings      # noqa: E402
from app.pipeline import Pipeline    # noqa: E402


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--image", required=True)
    ap.add_argument("--n", type=int, default=100)
    a = ap.parse_args()
    img = cv2.imread(a.image)
    p = Pipeline(settings)
    for _ in range(5): p.process(img)               # прогрев
    t = []
    for _ in range(a.n):
        s = time.perf_counter(); r = p.process(img); t.append((time.perf_counter() - s) * 1000)
    t = np.array(t)
    print(f"status={r['status']}  mean={t.mean():.1f} ms  p50={np.percentile(t,50):.1f}  "
          f"p95={np.percentile(t,95):.1f}  FPS~{1000/t.mean():.1f}")


if __name__ == "__main__":
    main()
