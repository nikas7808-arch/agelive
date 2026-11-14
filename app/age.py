"""Оценка возраста: DEX (101 класс) или прямая регрессия в ONNX."""
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
    low: float
    high: float
    confidence: float
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


def preprocess(bgr_crop: np.ndarray, size: int = 224) -> np.ndarray:
    img = cv2.resize(bgr_crop, (size, size))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    img = (img - _MEAN) / _STD
    return img.transpose(2, 0, 1)[None].astype(np.float32)


class AgeEstimator:
    def __init__(self, model_path: str, mock: bool = False) -> None:
        self.mock = mock
        self.sess = None
        self.size = 224
        if not mock:
            if not os.path.exists(model_path):
                raise FileNotFoundError(
                    f"Нет модели возраста: {model_path}. Обучите (scripts/train_age.py), "
                    "положите в models/ или запустите с AGELIVE_MOCK=1."
                )
            import onnxruntime as ort
            self.sess = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
            inp = self.sess.get_inputs()[0]
            self.inp = inp.name
            if len(inp.shape) == 4 and isinstance(inp.shape[-1], int) and inp.shape[-1] > 0:
                self.size = int(inp.shape[-1])

    def predict(self, bgr_crop: np.ndarray) -> AgeResult:
        if self.mock:
            g = np.exp(-0.5 * ((AGES - 30.0) / 6.0) ** 2)
            return distribution_stats(g)
        raw = self.sess.run(None, {self.inp: preprocess(bgr_crop, self.size)})[0][0].astype(np.float32)
        if raw.size == 101:
            return distribution_stats(softmax(raw))
        est = float(np.clip(raw[2] * 100.0 if raw.size == 3 else raw[0], 0.0, 100.0))
        g = np.exp(-0.5 * ((AGES - est) / 3.5) ** 2)
        return distribution_stats(g)
