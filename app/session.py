"""Сессия потока: буфер последних кадров для сглаживания + состояние challenge."""
import threading
import time
from collections import deque
from typing import Dict, Optional

import numpy as np


class Session:
    def __init__(self, window: int) -> None:
        self.ages = deque(maxlen=window)
        self.ranges = deque(maxlen=window)
        self.confs = deque(maxlen=window)
        self.lives = deque(maxlen=window)
        self.challenge = None
        self.verified_until = 0.0
        self.last_seen = time.time()

    def push_live(self, s: float) -> float:
        self.lives.append(s)
        return float(np.mean(self.lives))

    def push_age(self, age: float, rng, conf: float):
        self.ages.append(age); self.ranges.append(rng); self.confs.append(conf)
        return (float(np.median(self.ages)),
                list(np.median(np.array(self.ranges), axis=0)),
                float(np.mean(self.confs)))


class SessionStore:
    def __init__(self, window: int, ttl: float) -> None:
        self._s: Dict[str, Session] = {}
        self._lock = threading.Lock()
        self.window, self.ttl = window, ttl

    def get(self, sid: str) -> Session:
        now = time.time()
        with self._lock:
            for k in [k for k, v in self._s.items() if now - v.last_seen > self.ttl]:
                del self._s[k]
            s = self._s.setdefault(sid, Session(self.window))
            s.last_seen = now
            return s

    def drop(self, sid: str) -> None:
        with self._lock:
            self._s.pop(sid, None)
