"""Prometheus-метрики AgeLive. Отдаются на GET /metrics.

Правила, чтобы Prometheus не "взорвался":
- в лейблах только ограниченные наборы значений (никаких session_id, сырых URL, возрастов);
- гистограммы с фиксированными бакетами.
"""
from fastapi import Response
from prometheus_client import (CONTENT_TYPE_LATEST, Counter, Gauge, Histogram,
                               generate_latest)

_UNIT = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)
_FAST = (0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5)

# ─── HTTP ───
HTTP_REQUESTS = Counter(
    "agelive_http_requests_total", "HTTP-запросы", ["method", "endpoint", "status"])
HTTP_LATENCY = Histogram(
    "agelive_http_request_duration_seconds", "Время обработки HTTP-запроса",
    ["method", "endpoint"], buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0))

# ─── Пайплайн ───
# stage: detect | quality | passive_liveness | age
PIPE_STAGE_LATENCY = Histogram(
    "agelive_pipeline_stage_duration_seconds", "Время стадии пайплайна", ["stage"], buckets=_FAST)
# status: ok | no_face | bad_quality | challenge | uncertain | rejected
PIPE_OUTCOME = Counter(
    "agelive_pipeline_outcome_total", "Итоговый статус обработки кадра", ["status"])
# reason: no_face | face_too_small | blurry | too_dark | too_bright | head_turned |
#         spoof_suspected | challenge_failed | low_liveness
PIPE_REASON = Counter(
    "agelive_pipeline_reason_total", "Причины отказа/неуспеха", ["reason"])

# ─── Liveness ───
LIVENESS_SCORE = Histogram(
    "agelive_liveness_score", "Passive liveness score до сглаживания (0=спуф, 1=живой)",
    buckets=_UNIT)
# kind: blink | turn_left | turn_right
CHALLENGE_ISSUED = Counter(
    "agelive_challenge_issued_total", "Выданные active-challenge", ["kind"])
# result: passed | failed (failed включает таймаут)
CHALLENGE_RESULT = Counter(
    "agelive_challenge_result_total", "Результат active-challenge", ["kind", "result"])

# ─── Возраст ───
AGE_PREDICTION = Histogram(
    "agelive_age_prediction_years", "Предсказанный возраст (после сглаживания)",
    buckets=tuple(range(0, 101, 5)))
AGE_CONFIDENCE = Histogram(
    "agelive_age_confidence", "Уверенность оценки возраста", buckets=_UNIT)

# ─── WebSocket ───
WS_CONNECTIONS = Gauge("agelive_ws_connections_active", "Активные WebSocket-подключения")
# result: ok | invalid_image
WS_FRAMES = Counter("agelive_ws_frames_total", "Кадры, принятые по WebSocket", ["result"])

# ─── Состояние сервиса ───
# Одна серия со значением 1; по лейблам видно, какой бэкенд реально поднялся
# (liveness_backend="heuristic" значит, что antispoof.onnx не загрузился).
BACKEND_INFO = Gauge(
    "agelive_backend_info", "Какие бэкенды подняты при старте",
    ["detector", "liveness_backend", "mock"])


# Пороги из конфига: нужны дашборду, чтобы показать зоны отказ / челлендж / живой
LIVENESS_THRESHOLD = Gauge(
    "agelive_liveness_threshold", "Пороги liveness из конфига", ["name"])


def set_thresholds(cfg) -> None:
    LIVENESS_THRESHOLD.labels("t_spoof").set(cfg.t_spoof)
    LIVENESS_THRESHOLD.labels("t_live").set(cfg.t_live)


def set_backend_info(info: dict) -> None:
    BACKEND_INFO.clear()
    BACKEND_INFO.labels(
        detector=str(info.get("detector")),
        liveness_backend=str(info.get("liveness_backend")),
        mock=str(info.get("mock")).lower(),
    ).set(1)


def metrics_response() -> Response:
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
