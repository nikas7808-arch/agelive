# AgeLive: определение возраста по камере с liveness

Pipeline: детекция → quality gate → liveness (passive + active) → возраст → JSON. Возраст отдаётся только живым лицам.

## Быстрый старт
```bash
pip install -r requirements.txt
AGELIVE_MOCK=1 uvicorn app.main:app --reload     # без весов, заглушки моделей
# с весами: положите models/age.onnx (и models/antispoof.onnx), затем без MOCK
```
Демо с веб-камерой: http://localhost:8000/demo/ (камера требует HTTPS или localhost).

Docker: `docker build -t agelive . && docker run -p 8000:8000 agelive`

## API
- `POST /estimate?threshold=18&session_id=...` (multipart, поле `file`) → JSON
- `WS /stream?threshold=18` - бинарные JPEG-кадры туда, JSON обратно (challenge работает в потоке)
- `GET /health`

```json
{"status":"ok","live":true,"liveness_score":0.97,"age":34.2,"age_range":[30,38],
 "confidence":0.88,"p_over_threshold":0.99,"threshold":18,"latency_ms":62}
```
`status`: `ok | no_face | bad_quality | challenge | uncertain | rejected`.

## Модели
1. **Возраст:** `python scripts/train_age.py --data data/UTKFace --epochs 20` (EfficientNet-B0, DEX) → `models/age.onnx`
   (`pip install -r requirements-train.txt`). Или экспортируйте свою модель с входом 1×3×224×224 и выходом 101 логит.
2. **Liveness:** возьмите Silent-Face-Anti-Spoofing (MiniFASNetV2), экспортируйте в ONNX → `models/antispoof.onnx`
   (вход 1×3×80×80 BGR 0-255, класс 1 = real; проверьте на своей экспортированной модели).
   Без файла используется слабая эвристика (муар + блики), только для демо.

## Оценка
```bash
python scripts/eval_age.py --data data/UTKFace --list models/test_files.txt
python scripts/eval_liveness.py --data data/attacks      # live/ screen/ print/ ...
python scripts/benchmark.py --image face.jpg --n 100
pytest
```
Настройки порогов: переменные `AGELIVE_*` (см. `app/config.py`). Документы: `docs/`.

## Статус проверки
Логика (age, quality, active challenge, пайплайн) покрыта тестами. Сквозной запуск с реальными моделями,
MediaPipe и камерой нужно проверить на вашей машине: в среде, где собирался репозиторий, не было сети и FastAPI.
