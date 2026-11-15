# Метрики (Prometheus)

Сервис отдаёт метрики на `GET /metrics` в формате Prometheus. Код: `app/metrics.py`,
точки инструментации: `app/main.py` (HTTP, WebSocket) и `app/pipeline.py` (стадии, liveness, возраст).

## Зачем
Без метрик мы не знаем, сколько запросов держит сервис, какая стадия пайплайна самая медленная,
как часто срабатывает защита от спуфинга и поднялась ли модель при старте.

## Быстрая проверка
```bash
uvicorn app.main:app
curl http://localhost:8000/metrics
```
Для графиков: `monitoring/prometheus.yml` (команда запуска Prometheus в комментарии файла).

## Список метрик
| Метрика | Тип | Лейблы | Что показывает |
|---|---|---|---|
| `agelive_http_requests_total` | counter | method, endpoint, status | Запросы и коды ответов (RPS, доля ошибок) |
| `agelive_http_request_duration_seconds` | histogram | method, endpoint | Латентность HTTP |
| `agelive_pipeline_stage_duration_seconds` | histogram | stage: detect, quality, passive_liveness, age | Время каждой стадии, где узкое место |
| `agelive_pipeline_outcome_total` | counter | status: ok, no_face, bad_quality, challenge, uncertain, rejected | Чем заканчивается обработка кадра |
| `agelive_pipeline_reason_total` | counter | reason | Причины отказов (blurry, spoof_suspected, challenge_failed...) |
| `agelive_liveness_score` | histogram | - | Распределение passive liveness (сырой скор до сглаживания) |
| `agelive_challenge_issued_total` | counter | kind: blink, turn_left, turn_right | Сколько челленджей выдано |
| `agelive_challenge_result_total` | counter | kind, result: passed, failed | Исход челленджей (failed включает таймаут) |
| `agelive_age_prediction_years` | histogram | - | Распределение предсказанного возраста |
| `agelive_age_confidence` | histogram | - | Уверенность оценки возраста |
| `agelive_ws_connections_active` | gauge | - | Активные WebSocket-подключения |
| `agelive_ws_frames_total` | counter | result: ok, invalid_image | Кадры по WebSocket |
| `agelive_backend_info` | gauge | detector, liveness_backend, mock | Какие бэкенды подняты. `liveness_backend="heuristic"` = antispoof.onnx не загрузился |

`/metrics` сам себя в счётчиках не учитывает. Незнакомые URL попадают в `endpoint="unmatched"`.

## Полезные запросы (PromQL)
```promql
# запросов в секунду на /estimate
rate(agelive_http_requests_total{endpoint="/estimate"}[1m])

# p95 по стадиям пайплайна
histogram_quantile(0.95, sum by (stage, le) (rate(agelive_pipeline_stage_duration_seconds_bucket[5m])))

# доля отклонённых как спуф
sum(rate(agelive_pipeline_reason_total{reason="spoof_suspected"}[5m])) / sum(rate(agelive_pipeline_outcome_total[5m]))

# доля челленджей, которые не прошли (включая таймауты)
sum(rate(agelive_challenge_result_total{result="failed"}[10m])) / sum(rate(agelive_challenge_result_total[10m]))

# упал ли liveness-модель на эвристику
agelive_backend_info{liveness_backend="heuristic"}
```

## Чего здесь нет: FAR/FRR
FAR/FRR считаются по размеченным данным, а онлайн мы не знаем, кто был реальной атакой.
Их даёт офлайн-оценка: `python scripts/eval_liveness.py --data data/attacks`.
`agelive_liveness_score` помогает только косвенно: по форме распределения видно дрейф скоров.

## Правила для тех, кто добавляет новые метрики
- В лейблах только ограниченные наборы значений. Никаких `session_id`, сырых URL, возрастов.
- Метрика определяется один раз в `app/metrics.py`, наблюдается в одном месте (иначе двойной счёт).
