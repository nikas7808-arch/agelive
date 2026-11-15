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
from .metrics import (AGE_CONFIDENCE, AGE_PREDICTION, CHALLENGE_ISSUED, CHALLENGE_RESULT,
                      LIVENESS_SCORE, PIPE_OUTCOME, PIPE_REASON, PIPE_STAGE_LATENCY)
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
            PIPE_OUTCOME.labels(status).inc()
            for r in kw["reasons"]:
                PIPE_REASON.labels(r).inc()
            return {"status": status, "latency_ms": round((time.perf_counter() - t0) * 1000, 1), **kw}

        with PIPE_STAGE_LATENCY.labels("detect").time():
            face = self.detector.detect(frame)
        if face is None:
            return out("no_face", reasons=["no_face"])

        # 1) идёт active challenge
        just_passed = False
        if sess and sess.challenge is not None:
            st = sess.challenge.update(face, now)
            if st == "pending":
                return out("challenge", live=False, challenge=sess.challenge.kind)
            CHALLENGE_RESULT.labels(sess.challenge.kind, st).inc()
            sess.challenge = None
            if st == "failed":
                return out("rejected", live=False, reasons=["challenge_failed"])
            sess.verified_until = now + cfg.verified_ttl_s
            just_passed = True

        # 2) quality gate
        with PIPE_STAGE_LATENCY.labels("quality").time():
            q = assess(face, cfg, check_pose=not just_passed)
        if not q.ok:
            return out("bad_quality", reasons=q.reasons)

        # 3) passive liveness (+ сглаживание по сессии)
        with PIPE_STAGE_LATENCY.labels("passive_liveness").time():
            score = self.passive.score(frame, face)
        LIVENESS_SCORE.observe(score)
        if sess:
            score = sess.push_live(score)
        score = round(score, 3)
        if score < cfg.t_spoof:
            return out("rejected", live=False, liveness_score=score, reasons=["spoof_suspected"])
        verified = bool(sess and now < sess.verified_until)
        if score < cfg.t_live and not verified:
            if sess is not None and self.detector.supports_active:
                sess.challenge = ActiveChallenge.random(cfg.challenge_timeout_s, now)
                CHALLENGE_ISSUED.labels(sess.challenge.kind).inc()
                return out("challenge", live=False, liveness_score=score,
                           challenge=sess.challenge.kind)
            return out("uncertain", live=False, liveness_score=score, reasons=["low_liveness"])

        # 4) возраст
        with PIPE_STAGE_LATENCY.labels("age").time():
            res = self.age.predict(face.crop)
        age, rng, conf, probs = res.age, res.interval, res.confidence, res.probs
        if sess:
            age, rng, conf, probs = sess.push_age(age, rng, conf, probs)
        AGE_PREDICTION.observe(age)
        AGE_CONFIDENCE.observe(conf)
        p_over = float(probs[threshold:].sum())
        return out("ok", live=True, liveness_score=score, age=round(age, 1),
                   age_range=[round(float(v), 1) for v in rng], confidence=round(conf, 2),
                   p_over_threshold=round(p_over, 3), threshold=threshold)
