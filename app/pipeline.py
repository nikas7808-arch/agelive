"""Оркестратор: детекция -> quality gate -> liveness -> возраст.

Возраст возвращается ТОЛЬКО если liveness пройден.
"""
import time
from typing import Optional

import numpy as np

from .age import AgeEstimator
from .config import Settings
from .detector import FaceDetector
from .liveness.active import ActiveChallenge
from .liveness.passive import PassiveLiveness
from .quality import assess
from .session import SessionStore


class Pipeline:
    def __init__(self, cfg: Settings, detector=None) -> None:
        self.cfg = cfg
        self.detector = detector or FaceDetector()
        self.passive = PassiveLiveness(cfg.live_model, cfg.mock)
        self.age = AgeEstimator(cfg.age_model, cfg.mock)
        self.sessions = SessionStore(cfg.smooth_window, cfg.session_ttl_s)

    def info(self) -> dict:
        return {"detector": "mediapipe" if self.detector.supports_active else "haar",
                "liveness_backend": self.passive.backend,
                "active_challenge": self.detector.supports_active,
                "mock": self.cfg.mock}

    def process(self, frame: np.ndarray, session_id: Optional[str] = None,
                threshold: int = 18) -> dict:
        t0 = time.perf_counter()
        cfg = self.cfg
        now = time.time()
        sess = self.sessions.get(session_id) if session_id else None

        def out(status, **kw):
            kw.setdefault("reasons", [])
            return {"status": status, "latency_ms": round((time.perf_counter() - t0) * 1000, 1), **kw}

        face = self.detector.detect(frame)
        if face is None:
            return out("no_face", reasons=["no_face"])

        # 1) идёт active challenge
        if sess and sess.challenge is not None:
            st = sess.challenge.update(face, now)
            if st == "pending":
                return out("challenge", live=False, challenge=sess.challenge.kind)
            sess.challenge = None
            if st == "failed":
                return out("rejected", live=False, reasons=["challenge_failed"])
            sess.verified_until = now + cfg.verified_ttl_s

        # 2) quality gate
        q = assess(face, cfg, check_pose=True)
        if not q.ok:
            return out("bad_quality", reasons=q.reasons)

        # 3) passive liveness (+ сглаживание по сессии)
        score = self.passive.score(frame, face)
        if sess:
            score = sess.push_live(score)
        score = round(score, 3)
        if score < cfg.t_spoof:
            return out("rejected", live=False, liveness_score=score, reasons=["spoof_suspected"])
        verified = bool(sess and now < sess.verified_until)
        if score < cfg.t_live and not verified:
            if sess is not None and self.detector.supports_active:
                sess.challenge = ActiveChallenge.random(cfg.challenge_timeout_s, now)
                return out("challenge", live=False, liveness_score=score,
                           challenge=sess.challenge.kind)
            return out("uncertain", live=False, liveness_score=score, reasons=["low_liveness"])

        # 4) возраст
        res = self.age.predict(face.crop)
        age, rng, conf = res.age, res.interval, res.confidence
        if sess:
            age, rng, conf = sess.push_age(age, rng, conf)
        return out("ok", live=True, liveness_score=score, age=round(age, 1),
                   age_range=[round(float(v), 1) for v in rng], confidence=round(conf, 2),
                   p_over_threshold=round(res.p_over(threshold), 3), threshold=threshold)
