"""Passive liveness: оценка одного кадра без действий пользователя."""
import os
import cv2
import numpy as np


def _scaled_crop(frame: np.ndarray, bbox, scale: float = 2.7, size: int = 80) -> np.ndarray:
    x, y, w, h = bbox
    H, W = frame.shape[:2]
    nw, nh = max(w, 1) * scale, max(h, 1) * scale
    cx, cy = x + w / 2.0, y + h / 2.0
    x0, y0 = int(round(cx - nw / 2.0)), int(round(cy - nh / 2.0))
    x1, y1 = int(round(cx + nw / 2.0)), int(round(cy + nh / 2.0))
    pad_l, pad_t = max(0, -x0), max(0, -y0)
    pad_r, pad_b = max(0, x1 - W), max(0, y1 - H)
    crop = frame[max(0, y0):min(H, y1), max(0, x0):min(W, x1)]
    if crop.size == 0:
        return np.zeros((size, size, 3), dtype=np.uint8)
    if pad_l or pad_t or pad_r or pad_b:
        crop = cv2.copyMakeBorder(
            crop, pad_t, pad_b, pad_l, pad_r, cv2.BORDER_CONSTANT, value=(0, 0, 0)
        )
    return cv2.resize(crop, (size, size))


def moire_score(crop_bgr: np.ndarray) -> float:
    g = cv2.cvtColor(cv2.resize(crop_bgr, (128, 128)), cv2.COLOR_BGR2GRAY).astype(np.float32)
    f = np.log1p(np.abs(np.fft.fftshift(np.fft.fft2(g - g.mean()))))
    yy, xx = np.indices(f.shape)
    r = np.hypot(yy - 64, xx - 64)
    band = f[(r > 30) & (r < 62)]
    peak = float(band.max() / (np.median(band) + 1e-6))
    return float(np.clip((peak - 2.0) / 3.0, 0.0, 1.0))


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
        self.size = 80
        self.real_idx = int(os.getenv("AGELIVE_LIVE_REAL_IDX", "1"))
        if not mock and os.path.exists(model_path):
            import onnxruntime as ort
            self.sess = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
            inp = self.sess.get_inputs()[0]
            self.inp = inp.name
            if len(inp.shape) == 4 and isinstance(inp.shape[-1], int) and inp.shape[-1] > 0:
                self.size = int(inp.shape[-1])

    @property
    def backend(self) -> str:
        return "mock" if self.mock else ("onnx" if self.sess else "heuristic")

    def score(self, frame: np.ndarray, face) -> float:
        if self.mock:
            return 0.95
        if self.sess is None:
            return heuristic_score(face.crop)
        x = _scaled_crop(frame, face.bbox, size=self.size).astype(np.float32).transpose(2, 0, 1)[None]
        out = self.sess.run(None, {self.inp: x})[0][0].astype(np.float32)
        if out.size == 1:
            v = float(out[0])
            return v if 0.0 <= v <= 1.0 else float(1.0 / (1.0 + np.exp(-v)))
        if np.all(out >= 0.0) and np.all(out <= 1.0) and abs(float(out.sum()) - 1.0) < 1e-2:
            p = out
        else:
            e = np.exp(out - out.max())
            p = e / e.sum()
        idx = self.real_idx if 0 <= self.real_idx < p.size else 1
        return float(p[idx])
