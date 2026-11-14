# Результаты замеров

## 1. Оценка точности возраста (UTKFace test split)
```text
N=2362  MAE=4.68

MAE по возрастным группам:
    0-12  n=348   MAE=1.61
   13-17  n=89    MAE=4.63
   18-25  n=333   MAE=3.73
   26-40  n=929   MAE=4.37
   41-60  n=433   MAE=6.61
   61-120 n=230   MAE=8.29

MAE по полю 'пол':
  male    n=1242  MAE=5.01
  female  n=1120  MAE=4.30

MAE по полю 'раса':
  White   n=1025  MAE=4.95
  Black   n=439   MAE=5.55
  Asian   n=341   MAE=3.33
  Indian  n=401   MAE=4.60
  Other   n=156   MAE=3.55

Точность на порогах (pred>=T совпадает с true>=T):
  T=18: 0.968
  T=21: 0.950
```

## 2. Скорость работы пайплайна (Latency & FPS, MacBook M2)
```text
status=bad_quality  mean=2.7 ms  p50=2.7  p95=2.8  FPS~365.9
```

## 3. Антиспуфинг (APCER / BPCER)
Запускается командой `python scripts/eval_liveness.py --data data/attacks` на тестовых фото (`live/`, `screen/`, `print/`).
