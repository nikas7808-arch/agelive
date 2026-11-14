"""Quality gate: плохой кадр -> просим повторить, а не гадаем."""
from dataclasses import dataclass, field
from typing import Dict, List

import cv2

from .config import Settings
from .detector import Face


@dataclass
class Quality:
    ok: bool
    reasons: List[str] = field(default_factory=list)
    metrics: Dict[str, float] = field(default_factory=dict)


def assess(face: Face, cfg: Settings, check_pose: bool = True) -> Quality:
    gray = cv2.cvtColor(face.crop, cv2.COLOR_BGR2GRAY)
    blur = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    bright = float(gray.mean())
    size = min(face.bbox[2], face.bbox[3])
    reasons = []
    if size < cfg.min_face_px:
        reasons.append("face_too_small")
    if blur < cfg.min_blur_var:
        reasons.append("blurry")
    if bright < cfg.min_brightness:
        reasons.append("too_dark")
    if bright > cfg.max_brightness:
        reasons.append("too_bright")
    if check_pose and abs(face.yaw) > cfg.max_yaw:
        reasons.append("head_turned")
    return Quality(not reasons, reasons,
                   {"blur": blur, "brightness": bright, "face_px": size, "yaw": face.yaw})
