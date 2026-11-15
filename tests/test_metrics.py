import os
os.environ["AGELIVE_MOCK"] = "1"

import cv2
import numpy as np
from fastapi.testclient import TestClient
from prometheus_client import REGISTRY

from app.config import Settings
from app.detector import Face
from app.main import app
from app.pipeline import Pipeline

client = TestClient(app)


def val(name, **labels):
    return REGISTRY.get_sample_value(name, labels) or 0.0


class FakeDet:
    supports_active = True

    def __init__(self):
        self.face = None

    def detect(self, _):
        return self.face


def face():
    img = np.random.RandomState(0).randint(80, 180, (224, 224, 3)).astype(np.uint8)
    return Face((0, 0, 150, 150), img, 0.3, 0.0, True)


FRAME = np.zeros((240, 320, 3), np.uint8)


def test_metrics_endpoint_exposes_expected_series():
    client.get("/health")
    r = client.get("/metrics")
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/plain")
    for name in ("agelive_http_requests_total", "agelive_pipeline_stage_duration_seconds",
                 "agelive_liveness_score", "agelive_ws_connections_active", "agelive_backend_info"):
        assert name in r.text


def test_http_counter_uses_route_template_and_skips_metrics():
    before = val("agelive_http_requests_total", method="GET", endpoint="/health", status="200")
    client.get("/health")
    assert val("agelive_http_requests_total", method="GET", endpoint="/health", status="200") == before + 1
    # мусорные URL не плодят серии: все уходят в "unmatched"
    client.get("/no/such/path/123")
    assert val("agelive_http_requests_total", method="GET", endpoint="unmatched", status="404") >= 1
    assert "/no/such/path/123" not in client.get("/metrics").text
    # сам /metrics себя не считает
    assert 'endpoint="/metrics"' not in client.get("/metrics").text


def test_estimate_400_is_counted():
    before = val("agelive_http_requests_total", method="POST", endpoint="/estimate", status="400")
    client.post("/estimate", files={"file": ("x.jpg", b"not an image")})
    assert val("agelive_http_requests_total", method="POST", endpoint="/estimate", status="400") == before + 1


def test_pipeline_ok_updates_stage_age_and_liveness():
    p = Pipeline(Settings(mock=True), detector=FakeDet())
    p.detector.face = face()
    ok0 = val("agelive_pipeline_outcome_total", status="ok")
    age0 = val("agelive_pipeline_stage_duration_seconds_count", stage="age")
    live0 = val("agelive_liveness_score_count")
    r = p.process(FRAME, "m1")
    assert r["status"] == "ok"
    assert val("agelive_pipeline_outcome_total", status="ok") == ok0 + 1
    assert val("agelive_pipeline_stage_duration_seconds_count", stage="age") == age0 + 1
    assert val("agelive_liveness_score_count") == live0 + 1


def test_spoof_counts_reason_and_skips_age_stage():
    p = Pipeline(Settings(mock=True), detector=FakeDet())
    p.passive.score = lambda f, fc: 0.1
    p.detector.face = face()
    rej0 = val("agelive_pipeline_outcome_total", status="rejected")
    reason0 = val("agelive_pipeline_reason_total", reason="spoof_suspected")
    age0 = val("agelive_pipeline_stage_duration_seconds_count", stage="age")
    p.process(FRAME, "m2")
    assert val("agelive_pipeline_outcome_total", status="rejected") == rej0 + 1
    assert val("agelive_pipeline_reason_total", reason="spoof_suspected") == reason0 + 1
    assert val("agelive_pipeline_stage_duration_seconds_count", stage="age") == age0


def test_no_face_counted():
    p = Pipeline(Settings(mock=True), detector=FakeDet())
    n0 = val("agelive_pipeline_outcome_total", status="no_face")
    p.process(FRAME, "m3")
    assert val("agelive_pipeline_outcome_total", status="no_face") == n0 + 1


def test_challenge_issued_and_result():
    p = Pipeline(Settings(mock=True), detector=FakeDet())
    p.passive.score = lambda f, fc: 0.6        # серая зона: t_spoof < 0.6 < t_live
    p.detector.face = face()
    issued0 = sum(val("agelive_challenge_issued_total", kind=k) for k in ("blink", "turn_left", "turn_right"))
    r = p.process(FRAME, "m4")
    assert r["status"] == "challenge"
    issued1 = sum(val("agelive_challenge_issued_total", kind=k) for k in ("blink", "turn_left", "turn_right"))
    assert issued1 == issued0 + 1
    # просрочиваем челлендж -> failed
    kind = p.sessions.get("m4").challenge.kind
    p.sessions.get("m4").challenge.start -= 1000
    failed0 = val("agelive_challenge_result_total", kind=kind, result="failed")
    p.process(FRAME, "m4")
    assert val("agelive_challenge_result_total", kind=kind, result="failed") == failed0 + 1


def test_websocket_gauge_and_frames():
    base = val("agelive_ws_connections_active")
    frames0 = val("agelive_ws_frames_total", result="invalid_image")
    with client.websocket_connect("/stream") as ws:
        ws.send_bytes(b"garbage")
        assert ws.receive_json()["status"] == "error"
        assert val("agelive_ws_connections_active") == base + 1
        img = cv2.imencode(".jpg", np.zeros((120, 160, 3), np.uint8))[1].tobytes()
        ws.send_bytes(img)
        assert ws.receive_json()["status"] == "no_face"
    assert val("agelive_ws_connections_active") == base
    assert val("agelive_ws_frames_total", result="invalid_image") == frames0 + 1
