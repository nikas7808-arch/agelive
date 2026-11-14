"""Обучение модели возраста (DEX: CE по мягким меткам + 0.1*L1 по матожиданию).

Пример:
  python scripts/train_age.py --data data/UTKFace --epochs 20 --out models/age.onnx
Имя файла UTKFace: [age]_[gender]_[race]_[date].jpg
Сплит по хешу имени (у UTKFace нет id человека: дубли возможны, это ограничение датасета).
"""
import argparse, glob, os, zlib

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.utils.data import DataLoader, Dataset

MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)
AGES = torch.arange(101).float()


class Faces(Dataset):
    def __init__(self, files, train):
        self.files, self.train = files, train

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        f = self.files[i]
        age = min(int(os.path.basename(f).split("_")[0]), 100)
        img = cv2.resize(cv2.imread(f), (224, 224))
        if self.train:
            if np.random.rand() < 0.5: img = img[:, ::-1]
            img = cv2.convertScaleAbs(img, alpha=np.random.uniform(.7, 1.3), beta=np.random.uniform(-30, 30))
            if np.random.rand() < 0.3: img = cv2.GaussianBlur(img, (5, 5), 0)
            if np.random.rand() < 0.3:
                _, enc = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, int(np.random.randint(30, 90))])
                img = cv2.imdecode(enc, 1)
        x = (cv2.cvtColor(np.ascontiguousarray(img), cv2.COLOR_BGR2RGB).astype(np.float32) / 255 - MEAN) / STD
        return torch.from_numpy(x.transpose(2, 0, 1)), age


def soft_labels(age, sigma=2.0):
    d = torch.exp(-0.5 * ((AGES[None] - age[:, None].float()) / sigma) ** 2)
    return d / d.sum(1, keepdim=True)


def split(files):
    h = lambda f: zlib.crc32(os.path.basename(f).encode()) % 100
    return ([f for f in files if h(f) < 80], [f for f in files if 80 <= h(f) < 90],
            [f for f in files if h(f) >= 90])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True); ap.add_argument("--out", default="models/age.onnx")
    ap.add_argument("--epochs", type=int, default=20); ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-4)
    a = ap.parse_args()
    files = sorted(glob.glob(os.path.join(a.data, "*.jpg")))
    tr, va, te = split(files)
    print(f"train {len(tr)} val {len(va)} test {len(te)}")
    open("models/test_files.txt", "w").write("\n".join(te))   # для честной оценки

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    m = torchvision.models.efficientnet_b0(weights="IMAGENET1K_V1")
    m.classifier[1] = nn.Linear(m.classifier[1].in_features, 101)
    m.to(dev)
    opt = torch.optim.AdamW(m.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.epochs)
    dl = DataLoader(Faces(tr, True), a.bs, shuffle=True, num_workers=4, drop_last=True)
    vl = DataLoader(Faces(va, False), a.bs, num_workers=4)
    best = 1e9
    for ep in range(a.epochs):
        m.train()
        for x, y in dl:
            x, y = x.to(dev), y.to(dev)
            lg = m(x)
            ce = -(soft_labels(y).to(dev) * F.log_softmax(lg, 1)).sum(1).mean()
            exp = (F.softmax(lg, 1) * AGES.to(dev)).sum(1)
            loss = ce + 0.1 * (exp - y.float()).abs().mean()
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
        m.eval(); err = []
        with torch.no_grad():
            for x, y in vl:
                e = (F.softmax(m(x.to(dev)), 1) * AGES.to(dev)).sum(1).cpu()
                err.append((e - y.float()).abs())
        mae = torch.cat(err).mean().item()
        print(f"epoch {ep+1}: val MAE {mae:.2f}")
        if mae < best:
            best = mae
            m.eval().cpu()
            torch.onnx.export(m, torch.zeros(1, 3, 224, 224), a.out, input_names=["input"],
                              output_names=["logits"], dynamic_axes={"input": {0: "b"}}, opset_version=17)
            m.to(dev)
    print("best val MAE", best, "->", a.out)


if __name__ == "__main__":
    main()
