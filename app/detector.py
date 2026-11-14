"""Детекция лица, кроп 224x224 и сигналы для active liveness (EAR, yaw).

Основной путь: MediaPipe FaceMesh (468 точек). Если MediaPipe нет, используется
Haar-каскад из OpenCV (только bbox, без EAR/yaw, active challenge недоступен).
"""
from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np

LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
CROP = 224


@dataclass
class Face:
    bbox: Tuple[int, int, int, int]      # x, y, w, h
    crop: np.ndarray                     # 224x224 BGR
    ear: Optional[float] = None          # eye aspect ratio
    yaw: float = 0.0                     # прокси поворота: <0 вправо, >0 влево (в сыром кадре)
    has_landmarks: bool = False


def _ear(pts: np.ndarray, idx) -> float:
    p = pts[idx]
    v = np.linalg.norm(p[1] - p[5]) + np.linalg.norm(p[2] - p[4])
    h = np.linalg.norm(p[0] - p[3]) + 1e-6
    return float(v / (2.0 * h))


def _crop(frame: np.ndarray, bbox, margin: float = 0.2) -> np.ndarray:
    x, y, w, h = bbox
    H, W = frame.shape[:2]
    m = int(max(w, h) * margin)
    x0, y0 = max(0, x - m), max(0, y - m)
    x1, y1 = min(W, x + w + m), min(H, y + h + m)
    return cv2.resize(frame[y0:y1, x0:x1], (CROP, CROP))


class FaceDetector:
    def __init__(self) -> None:
        self.mesh = None
        try:
            import mediapipe as mp
            self.mesh = mp.solutions.face_mesh.FaceMesh(
                static_image_mode=False, max_num_faces=1, refine_landmarks=True,
                min_detection_confidence=0.5, min_tracking_confidence=0.5)
        except Exception:
            self.mesh = None
        self.haar = None
        try:
            self.haar = cv2.CascadeClassifier(
                cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        except Exception:
            pass

    @property
    def supports_active(self) -> bool:
        return self.mesh is not None

    def detect(self, bgr: np.ndarray) -> Optional[Face]:
        if self.mesh is not None:
            return self._detect_mesh(bgr)
        return self._detect_haar(bgr)

    def _detect_mesh(self, bgr):
        H, W = bgr.shape[:2]
        res = self.mesh.process(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        if not res.multi_face_landmarks:
            return None
        lm = res.multi_face_landmarks[0].landmark
        pts = np.array([[p.x * W, p.y * H] for p in lm], dtype=np.float32)
        x0, y0 = pts.min(0)
        x1, y1 = pts.max(0)
        bbox = (int(x0), int(y0), int(x1 - x0), int(y1 - y0))
        ear = (_ear(pts, LEFT_EYE) + _ear(pts, RIGHT_EYE)) / 2.0
        mid_x = (pts[234][0] + pts[454][0]) / 2.0
        half_w = abs(pts[454][0] - pts[234][0]) / 2.0 + 1e-6
        yaw = float((pts[1][0] - mid_x) / half_w)
        return Face(bbox, _crop(bgr, bbox), ear, yaw, True)

    def _detect_haar(self, bgr):
        if self.haar is None or self.haar.empty():
            return None
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        faces = self.haar.detectMultiScale(gray, 1.2, 5, minSize=(60, 60))
        if len(faces) == 0:
            return None
        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
        bbox = (int(x), int(y), int(w), int(h))
        return Face(bbox, _crop(bgr, bbox))
