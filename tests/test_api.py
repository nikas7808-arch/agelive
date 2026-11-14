import os
os.environ["AGELIVE_MOCK"] = "1"

import cv2
import numpy as np
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def jpeg(img):
    return cv2.imencode(".jpg", img)[1].tobytes()


def test_health():
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["mock"] is True


def test_invalid_image():
    r = client.post("/estimate", files={"file": ("x.jpg", b"not an image")})
    assert r.status_code == 400


def test_blank_image_no_face():
    r = client.post("/estimate", files={"file": ("x.jpg", jpeg(np.zeros((240, 320, 3), np.uint8)))})
    assert r.status_code == 200
    assert r.json()["status"] == "no_face"
    assert r.json()["age"] is None      # без лица возраст не возвращается
