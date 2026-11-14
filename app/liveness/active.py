"""Active challenge: случайная команда (моргнуть / повернуть голову) с таймаутом.

ВАЖНО: активная проверка добавляет защиту от фото и статичных масок, но НЕ от
видео-повтора и deepfake: её нужно сочетать с passive liveness.
Для моргания нужна частота кадров >= 10 FPS.
"""
import random
from typing import Optional

EAR_CLOSED = 0.20
EAR_OPEN = 0.23
YAW_TURN = 0.25
KINDS = ("blink", "turn_left", "turn_right")


class ActiveChallenge:
    def __init__(self, kind: str, timeout: float, now: float) -> None:
        assert kind in KINDS
        self.kind = kind
        self.timeout = timeout
        self.start = now
        self.closed_frames = 0

    @classmethod
    def random(cls, timeout: float, now: float, rng: Optional[random.Random] = None):
        return cls((rng or random).choice(KINDS), timeout, now)

    def update(self, face, now: float) -> str:
        """Возвращает pending | passed | failed."""
        if now - self.start > self.timeout:
            return "failed"
        if not face.has_landmarks or face.ear is None:
            return "pending"
        if self.kind == "blink":
            if face.ear < EAR_CLOSED:
                self.closed_frames += 1
            elif face.ear > EAR_OPEN:
                if self.closed_frames >= 1:
                    return "passed"
                self.closed_frames = 0
        elif self.kind == "turn_right":      # пользователь поворачивает голову вправо
            if face.yaw < -YAW_TURN:
                return "passed"
        elif self.kind == "turn_left":
            if face.yaw > YAW_TURN:
                return "passed"
        return "pending"
