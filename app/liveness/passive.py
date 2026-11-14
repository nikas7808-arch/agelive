"""Passive liveness: оценка одного кадра без действий пользователя.

1) Если есть ONNX-модель (MiniFASNet из Silent-Face-Anti-Spoofing) - используем её.
2) Иначе эвристика (муар в FFT + блики). Это слабая БАЗОВАЯ линия для демо,
   не защита для продакшена.
Возвращает score в [0, 1]: 1 = живое лицо.
"""
import os
from typing import Tuple

import cv2
import numpy as np


def _scaled_crop(frame, bbox, scale=2.7, size=80):
    x, y, w, h = bbox
    H, W = frame.shape[:2]
    s = min((H - 1) / max(h, 1), (W - 1) / max(w, 1), scale)
    nw, nh = w * s, h * s
    cx, cy = x + w / 2, y + h / 2
    x0 = int(max(0, cx - nw / 2)); y0 = int(max(0, cy - nh / 2))
    x1 = int(min(W - 1, cx + nw / 2)); y1 = int(min(H - 1, cy + nh / 2))
    return cv2.resize(frame[y0:y1, x0:x1], (size, size))


def moire_score(crop_bgr: np.ndarray) -> float:
    """Доля 'спикулярности' в высоких частотах спектра: у экранов выше из-за пиксельной сетки."""
    g = cv2.cvtColor(cv2.resize(crop_bgr, (128, 128)), cv2.COLOR_BGR2GRAY).astype(np.float32)
    f = np.log1p(np.abs(np.fft.fftshift(np.fft.fft2(g - g.mean()))))
    yy, xx = np.indices(f.shape)
    r = np.hypot(yy - 64, xx - 64)
    band = f[(r > 30) & (r < 62)]
    peak = float(band.max() / (np.median(band) + 1e-6))
    return float(np.clip((peak - 2.0) / 3.0, 0.0, 1.0))   # 0 = нет муара, 1 = явный


def specular_ratio(crop_bgr: np.ndarray) -> float:
    g = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    return float((g > 250).mean())


def heuristic_score(crop_bgr: np.ndarray) -> float:
    spoof = 0.6 * moire_score(crop_bgr) + 0.4 * float(np.clip(specular_ratio(crop_bgr) / 0.03, 0, 1))
    return float(1.0 - spoof)


class PassiveLiveness:
    def __init__(self, model_path: str, mock: bool = False) -> None:
        self.mock = mock
        self.sess = None
        if not mock and os.path.exists(model_path):
            import onnxruntime as ort
            self.sess = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
            self.inp = self.sess.get_inputs()[0].name

    @property
    def backend(self) -> str:
        return "mock" if self.mock else ("onnx" if self.sess else "heuristic")

    def score(self, frame: np.ndarray, face) -> float:
        if self.mock:
            return 0.95
        if self.sess is None:
            return heuristic_score(face.crop)
        x = _scaled_crop(frame, face.bbox).astype(np.float32).transpose(2, 0, 1)[None]
        out = self.sess.run(None, {self.inp: x})[0][0]
        e = np.exp(out - out.max())
        p = e / e.sum()
        return float(p[1])     # класс 1 = real (проверьте для своей экспортированной модели)
