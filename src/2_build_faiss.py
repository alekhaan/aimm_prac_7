import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import faiss

DATA_DIR = Path("../data/hse_faces_miem")
OUT_DIR  = Path("../data/artifacts")
OUT_DIR.mkdir(exist_ok=True)

YUNET = "../models/yunet/face_detection_yunet_2023mar.onnx"
SFACE = "../models/sfacenet/face_recognition_sface_2021dec.onnx"

def l2norm(x):
    return x / (np.linalg.norm(x) + 1e-12)

# Проверки
print("[INFO] DATA_DIR:", DATA_DIR.resolve())
assert DATA_DIR.exists(), "Dataset dir not found"
imgs = []
for ext in ("*.jpg","*.JPG","*.jpeg","*.png"):
    imgs += list(DATA_DIR.glob(ext))
print(f"[INFO] Found {len(imgs)} images")

# Модели (понизили порог детекции — так надёжнее)
detector = cv2.FaceDetectorYN.create(YUNET, "", (320,320), 0.6, 0.3, 5000)
recognizer = cv2.FaceRecognizerSF.create(SFACE, "")

embeddings, meta = [], []

for p in tqdm(imgs):
    img = cv2.imread(str(p))
    if img is None:
        print("[WARN] read fail:", p.name); continue

    h, w = img.shape[:2]
    detector.setInputSize((w, h))
    _, dets = detector.detect(img)
    if dets is None or len(dets) == 0:
        print("[WARN] no face:", p.name); continue

    d = max(dets, key=lambda x: x[4])
    face = recognizer.alignCrop(img, d)
    feat = recognizer.feature(face).reshape(-1)
    feat = l2norm(feat).astype("float32")

    embeddings.append(feat)
    meta.append({"filename": p.name, "path": str(p)})

print(f"[INFO] Extracted {len(embeddings)} embeddings")
assert len(embeddings) > 0, "No faces extracted"

X = np.vstack(embeddings)
index = faiss.IndexFlatIP(X.shape[1])
index.add(X)

faiss.write_index(index, str(OUT_DIR / "faiss.index"))
pd.DataFrame(meta).to_csv(OUT_DIR / "meta.csv", index=False)
print(f"[DONE] FAISS built with {len(X)} faces")
