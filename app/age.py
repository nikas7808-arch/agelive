"""Оценка возраста: DEX-подход (классификация 0..100 + матожидание).

Модель: ONNX, вход 1x3x224x224 (RGB, ImageNet-нормализация), выход 1x101 логитов.
Обучение и экспорт: scripts/train_age.py.
"""
import os
from dataclasses import dataclass
from typing import List

import cv2
import numpy as np

AGES = np.arange(101, dtype=np.float32)
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


@dataclass
class AgeResult:
    age: float
    low: float          # 10-й перцентиль
    high: float         # 90-й перцентиль
    confidence: float   # масса вероятности в окне +-5 лет от среднего
    probs: np.ndarray

    def p_over(self, threshold: int) -> float:
        return float(self.probs[threshold:].sum())

    @property
    def interval(self) -> List[float]:
        return [round(self.low, 1), round(self.high, 1)]


def softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max())
    return e / e.sum()


def distribution_stats(probs: np.ndarray) -> AgeResult:
    probs = probs / probs.sum()
    mean = float((probs * AGES).sum())
    cdf = np.cumsum(probs)
    low = float(np.searchsorted(cdf, 0.10))
    high = float(np.searchsorted(cdf, 0.90))
    lo, hi = int(max(0, mean - 5)), int(min(100, mean + 5))
    conf = float(probs[lo:hi + 1].sum())
    return AgeResult(mean, low, high, conf, probs)


def preprocess(bgr_crop: np.ndarray) -> np.ndarray:
    img = cv2.resize(bgr_crop, (224, 224))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    img = (img - _MEAN) / _STD
    return img.transpose(2, 0, 1)[None].astype(np.float32)


class AgeEstimator:
    def __init__(self, model_path: str, mock: bool = False) -> None:
        self.mock = mock
        self.sess = None
        if not mock:
            if not os.path.exists(model_path):
                raise FileNotFoundError(
                    f"Нет модели возраста: {model_path}. Обучите (scripts/train_age.py), "
                    "положите в models/ или запустите с AGELIVE_MOCK=1.")
            import onnxruntime as ort
            self.sess = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
            self.inp = self.sess.get_inputs()[0].name

    def predict(self, bgr_crop: np.ndarray) -> AgeResult:
        if self.mock:
            g = np.exp(-0.5 * ((AGES - 30.0) / 6.0) ** 2)
            return distribution_stats(g)
        logits = self.sess.run(None, {self.inp: preprocess(bgr_crop)})[0][0]
        return distribution_stats(softmax(logits.astype(np.float32)))
