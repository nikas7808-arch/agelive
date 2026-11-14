from app.detector import Face
from app.liveness.active import ActiveChallenge
import numpy as np


def face(ear=0.3, yaw=0.0):
    return Face((0, 0, 100, 100), np.zeros((224, 224, 3), np.uint8), ear, yaw, True)


def test_blink_passes():
    c = ActiveChallenge("blink", 5, 0)
    assert c.update(face(0.3), 0.1) == "pending"
    assert c.update(face(0.1), 0.2) == "pending"
    assert c.update(face(0.3), 0.3) == "passed"


def test_no_blink_times_out():
    c = ActiveChallenge("blink", 5, 0)
    assert c.update(face(0.3), 6) == "failed"


def test_turn_right():
    c = ActiveChallenge("turn_right", 5, 0)
    assert c.update(face(yaw=0.0), 1) == "pending"
    assert c.update(face(yaw=-0.4), 2) == "passed"


def test_wrong_direction_does_not_pass():
    c = ActiveChallenge("turn_left", 5, 0)
    assert c.update(face(yaw=-0.5), 1) == "pending"
