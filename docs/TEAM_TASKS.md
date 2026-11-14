# Распределение на 5 человек (2 дня)

| # | Роль | Файлы | День 1 | День 2 |
|---|---|---|---|---|
| 1 | Тимлид | `app/pipeline.py`, `docs/` | контракты, оркестратор, скелет слайдов | презентация, README, питч |
| 2 | Возраст | `app/age.py`, `scripts/train_age.py`, `scripts/eval_age.py` | обучение/готовая модель, MAE | пороги 18/21, интервал, таблицы, fairness по группам |
| 3 | Liveness passive | `app/liveness/passive.py`, `scripts/eval_liveness.py` | модель MiniFASNet → ONNX, съёмка 20-30 атак | APCER/BPCER, подбор T_live/T_spoof |
| 4 | CV | `app/detector.py`, `app/quality.py`, `app/liveness/active.py` | детекция, quality gate | active challenge, тюнинг порогов EAR/yaw |
| 5 | Backend | `app/main.py`, `web/`, `Dockerfile`, `scripts/benchmark.py` | API, Docker, демо-страница | latency (mean/p95), запись демо-видео |

Definition of done: `pytest` зелёный; `docker build` проходит; демо: живой проходит, фото с экрана и распечатка отклоняются.
Заморозка кода: 18:00 день 1 (интеграция), 19:00 день 2 (сдача).
