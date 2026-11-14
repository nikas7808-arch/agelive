"""Проверка ONNX-весов в папке models/ перед запуском сервиса.

Использование:
  python scripts/check_models.py
  python scripts/check_models.py --image path/to/live_face.jpg
"""
import argparse
import os
import sys
import numpy as np
import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.config import settings
from app.age import AgeEstimator
from app.liveness.passive import PassiveLiveness
from app.detector import Face


def inspect_onnx(path: str) -> None:
    import onnxruntime as ort
    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    inp = sess.get_inputs()[0]
    out = sess.get_outputs()[0]
    print(f"[OK] {path}: input={inp.name} {inp.shape} ({inp.type}) -> output={out.name} {out.shape}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", help="Тестовое фото живого лица для проверки индекса класса liveness")
    args = ap.parse_args()

    img = cv2.imread(args.image) if args.image else np.random.randint(60, 200, (480, 640, 3), dtype=np.uint8)
    face = Face((160, 80, 240, 240), cv2.resize(img, (224, 224)), 0.3, 0.0, True)

    if os.path.exists(settings.age_model):
        inspect_onnx(settings.age_model)
        est = AgeEstimator(settings.age_model, mock=False)
        res = est.predict(face.crop)
        print(f"     Тест age: {res.age:.1f} лет, интервал={res.interval}, p(>=18)={res.p_over(18):.3f}")
    else:
        print(f"[MISSING] {settings.age_model} — положите обученную или скачанную модель возраста")

    if os.path.exists(settings.live_model):
        inspect_onnx(settings.live_model)
        pas = PassiveLiveness(settings.live_model, mock=False)
        sc = pas.score(img, face)
        print(f"     Тест liveness (real_idx={pas.real_idx}): score={sc:.3f}")
        if args.image and sc < 0.5:
            print("     ВНИМАНИЕ: на живом фото score < 0.5! Попробуйте запустить с AGELIVE_LIVE_REAL_IDX=0")
    else:
        print(f"[MISSING] {settings.live_model} — сейчас работает только базовый эвристический детектор")


if __name__ == "__main__":
    main()
