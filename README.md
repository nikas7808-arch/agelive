# AgeLive: оценка возраста по камере с защитой от спуфинга (Liveness)

Автономный сервис верификации возраста в реальном времени на базе **FastAPI**, **MediaPipe** и **ONNX Runtime**.
Возраст возвращается клиенту **только после прохождения проверки живости лица (Liveness)**.

## Требования
- **Python 3.11 или 3.12** (для работы MediaPipe FaceMesh и активных проверок моргания/поворота головы).
- Обученные веса уже лежат в `models/age.onnx` и `models/antispoof.onnx`.

## Быстрый старт (macOS / Linux)
```bash
git clone https://github.com/nikas7808-arch/agelive.git
cd agelive
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

## Быстрый старт (Windows PowerShell)
```powershell
git clone https://github.com/nikas7808-arch/agelive.git
cd agelive
py -3.12 -m venv .venv
.venv\Scripts\pip install -r requirements.txt
.venv\Scripts\uvicorn app.main:app --reload
```

Откройте в браузере **http://localhost:8000/demo/** и нажмите **«Старт»**.

## Проверка и замеры метрик
```bash
pytest
python scripts/check_models.py
python scripts/eval_age.py --data data/UTKFace --list models/test_files.txt
python scripts/eval_liveness.py --data data/attacks
python scripts/benchmark.py --image face.jpg --n 100
```

## Метрики (Prometheus)
Живой дашборд: **http://localhost:8000/dashboard** (без Docker, обновляется каждые 2 с). Сырой текст для Prometheus: **http://localhost:8000/metrics**: RPS, время стадий пайплайна, исходы и причины отказов,
liveness-скоры, результаты челленджей, WebSocket-подключения, состояние бэкендов.
Список и примеры запросов: [docs/METRICS.md](docs/METRICS.md). Конфиг Prometheus: `monitoring/prometheus.yml`.
