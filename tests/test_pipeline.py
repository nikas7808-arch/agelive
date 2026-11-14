import numpy as np
from app.config import Settings
from app.detector import Face
from app.pipeline import Pipeline

CFG = Settings(mock=True)


class FakeDet:
    supports_active = True

    def __init__(self):
        self.face = None

    def detect(self, _):
        return self.face


def textured_face(ear=0.3, yaw=0.0):
    img = np.random.RandomState(0).randint(80, 180, (224, 224, 3)).astype(np.uint8)
    return Face((0, 0, 150, 150), img, ear, yaw, True)


def make(score):
    p = Pipeline(CFG, detector=FakeDet())
    p.passive.score = lambda f, face: score
    return p


FRAME = np.zeros((240, 320, 3), np.uint8)


def test_live_returns_age():
    p = make(0.95); p.detector.face = textured_face()
    r = p.process(FRAME, "s1")
    assert r["status"] == "ok" and r["live"] and 25 < r["age"] < 35


def test_spoof_never_returns_age():
    p = make(0.1); p.detector.face = textured_face()
    r = p.process(FRAME, "s1")
    assert r["status"] == "rejected" and "age" not in r


def test_gray_zone_starts_challenge_then_passes():
    p = make(0.6); p.detector.face = textured_face()
    r = p.process(FRAME, "s1")
    assert r["status"] == "challenge"
    kind = r["challenge"]
    p.detector.face = textured_face(ear=0.1, yaw=-0.5 if kind == "turn_right" else 0.5 if kind == "turn_left" else 0.0)
    r = p.process(FRAME, "s1")
    if kind == "blink":
        assert r["status"] == "challenge"
        p.detector.face = textured_face(ear=0.3)
        r = p.process(FRAME, "s1")
    assert r["status"] == "ok" and r["age"] is not None


def test_gray_zone_without_session_is_uncertain():
    p = make(0.6); p.detector.face = textured_face()
    assert p.process(FRAME)["status"] == "uncertain"
