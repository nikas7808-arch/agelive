# Архитектура

```
Камера ─► FaceDetector (MediaPipe FaceMesh / Haar fallback)
            │ кроп 224×224, EAR, yaw
            ▼
        Quality gate (размер, blur, свет, поза) ──► bad_quality
            ▼
        Passive liveness (MiniFASNet ONNX / эвристика) ──► score < T_spoof ─► rejected
            │ T_spoof ≤ score < T_live
            ▼
        Active challenge (blink / turn_left / turn_right, таймаут) ─► challenge / rejected
            │ score ≥ T_live или challenge пройден
            ▼
        Age (EfficientNet-B0, DEX: 101 класс) ─► age, интервал P10-P90, confidence, p(≥18)
            ▼
        Сглаживание по сессии (медиана последних N кадров) ─► JSON
```

## Решения и обоснование
- **Liveness перед возрастом.** Для спуфа возраст не возвращается вообще.
- **DEX (классификация + матожидание)** стабильнее прямой регрессии и даёт распределение,
  из которого получаем интервал и вероятность «старше порога».
- **Серая зона по порогам.** Active challenge запускается только при неуверенности passive: UX не страдает.
- **Приватность.** Кадры не сохраняются; сессия хранит только числа и удаляется при отключении/TTL.
- **Stateless `/estimate`** без `session_id`: серая зона даёт `uncertain`, challenge недоступен.

## Известные ограничения (честно в отчёт)
- Эвристический passive liveness без модели слабый; для результатов нужна обученная модель.
- Active challenge не защищает от видео-повтора и deepfake сам по себе.
- `yaw`: прокси по положению носа, не точная поза (solvePnP - следующий шаг).
- Haar fallback не поддерживает active challenge.
- UTKFace: шумные метки, нет id людей, перекос по возрасту/расе.

## Следующие шаги
ONNX FP16/INT8 и TensorRT, батчинг; rPPG и детектор deepfake; fairness-аудит по полу/расе/возрасту;
кросс-датасетная оценка liveness (CelebA-Spoof → CASIA-FASD/OULU-NPU); нагрузочные тесты.
