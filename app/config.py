"""Настройки сервиса. Все значения переопределяются переменными окружения."""
import os
from dataclasses import dataclass


def _f(name: str, default: float) -> float:
    return float(os.getenv(name, default))


@dataclass(frozen=True)
class Settings:
    age_model: str = os.getenv("AGELIVE_AGE_MODEL", "models/age.onnx")
    live_model: str = os.getenv("AGELIVE_LIVE_MODEL", "models/antispoof.onnx")
    # MOCK=1: заглушки моделей (для тестов и разработки без весов)
    mock: bool = os.getenv("AGELIVE_MOCK", "0") == "1"

    # Liveness: score >= t_live -> живой; < t_spoof -> атака; между -> active challenge
    t_live: float = _f("AGELIVE_T_LIVE", 0.80)
    t_spoof: float = _f("AGELIVE_T_SPOOF", 0.40)
    challenge_timeout_s: float = _f("AGELIVE_CHALLENGE_TIMEOUT", 8.0)
    verified_ttl_s: float = _f("AGELIVE_VERIFIED_TTL", 30.0)

    # Quality gate
    min_face_px: int = int(_f("AGELIVE_MIN_FACE", 80))
    min_blur_var: float = _f("AGELIVE_MIN_BLUR", 40.0)
    min_brightness: float = _f("AGELIVE_MIN_BRIGHT", 50.0)
    max_brightness: float = _f("AGELIVE_MAX_BRIGHT", 220.0)
    max_yaw: float = _f("AGELIVE_MAX_YAW", 0.35)

    # Временное сглаживание
    smooth_window: int = int(_f("AGELIVE_SMOOTH_N", 7))
    session_ttl_s: float = _f("AGELIVE_SESSION_TTL", 60.0)
    max_upload_bytes: int = 5 * 1024 * 1024


settings = Settings()
