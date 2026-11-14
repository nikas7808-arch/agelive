# Результаты (заполнить после прогонов)

| Метрика | Значение | Как получено |
|---|---|---|
| MAE (UTKFace test) | | `scripts/eval_age.py --list models/test_files.txt` |
| Точность на пороге 18 / 21 | | то же |
| MAE по полу / расе / возрасту | | то же |
| APCER (экран / печать) | | `scripts/eval_liveness.py` |
| BPCER | | то же |
| Latency mean / p95, FPS | | `scripts/benchmark.py` |
