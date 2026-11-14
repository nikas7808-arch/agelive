import numpy as np
from app.config import settings
from app.detector import Face
from app.quality import assess


def mk(img, size=150, yaw=0.0):
    return Face((0, 0, size, size), img, 0.3, yaw, True)


def test_dark_and_blurry_rejected():
    q = assess(mk(np.zeros((224, 224, 3), np.uint8)), settings)
    assert not q.ok and "too_dark" in q.reasons and "blurry" in q.reasons


def test_sharp_normal_ok():
    rng = np.random.RandomState(0)
    img = rng.randint(80, 180, (224, 224, 3)).astype(np.uint8)
    assert assess(mk(img), settings).ok


def test_small_face_and_pose():
    rng = np.random.RandomState(1)
    img = rng.randint(80, 180, (224, 224, 3)).astype(np.uint8)
    q = assess(mk(img, size=40, yaw=0.6), settings)
    assert "face_too_small" in q.reasons and "head_turned" in q.reasons
    assert assess(mk(img, yaw=0.6), settings, check_pose=False).ok
