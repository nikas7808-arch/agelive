"""Оценка passive liveness на собственных данных.

Структура: data/attacks/live/*.jpg  data/attacks/screen/*.jpg  data/attacks/print/*.jpg ...
(папка 'live' - живые, остальные - атаки).  python scripts/eval_liveness.py --data data/attacks
Считает APCER (атака принята), BPCER (живой отклонён), ACER и accuracy при порогах из конфига.
"""
import argparse, glob, os, sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.config import settings                    # noqa: E402
from app.detector import FaceDetector              # noqa: E402
from app.liveness.passive import PassiveLiveness   # noqa: E402


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--data", required=True)
    ap.add_argument("--thr", type=float, default=settings.t_live)
    a = ap.parse_args()
    det, pas = FaceDetector(), PassiveLiveness(settings.live_model, settings.mock)
    print("backend:", pas.backend)
    scores = {}
    for d in sorted(os.listdir(a.data)):
        for f in glob.glob(os.path.join(a.data, d, "*")):
            img = cv2.imread(f)
            face = det.detect(img) if img is not None else None
            if face is not None:
                scores.setdefault(d, []).append(pas.score(img, face))
    live = np.array(scores.get("live", []))
    if len(live) == 0:
        sys.exit("нет живых примеров в папке 'live'")
    bpcer = float((live < a.thr).mean())
    print(f"live: n={len(live)}  BPCER={bpcer:.3f}")
    ap_list = []
    for k, v in scores.items():
        if k == "live": continue
        v = np.array(v); r = float((v >= a.thr).mean()); ap_list.append(r)
        print(f"{k}: n={len(v)}  APCER={r:.3f}")
    apcer = max(ap_list) if ap_list else float("nan")
    print(f"\nthr={a.thr}  APCER(worst)={apcer:.3f}  BPCER={bpcer:.3f}  ACER={(apcer + bpcer) / 2:.3f}")


if __name__ == "__main__":
    main()
