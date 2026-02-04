import cv2
import numpy as np
import pandas as pd
import faiss
import sys
from pathlib import Path

ART = Path("../data/artifacts")
YUNET = "../models/yunet/face_detection_yunet_2023mar.onnx"
SFACE = "../models/sfacenet/face_recognition_sface_2021dec.onnx"

def l2norm(x):
    return x / (np.linalg.norm(x) + 1e-12)

if len(sys.argv) < 2:
    print("Usage: python3 2_search_faiss_sface.py <query_image>")
    sys.exit(1)

img_path = sys.argv[1]
img = cv2.imread(img_path)
assert img is not None, "Cannot read query image"

detector = cv2.FaceDetectorYN.create(YUNET, "", (320,320), 0.6, 0.3, 5000)
recognizer = cv2.FaceRecognizerSF.create(SFACE, "")

h, w = img.shape[:2]
detector.setInputSize((w, h))
_, dets = detector.detect(img)
assert dets is not None and len(dets) > 0, "No face in query image"

d = max(dets, key=lambda x: x[4])
face = recognizer.alignCrop(img, d)
feat = l2norm(recognizer.feature(face).reshape(-1)).astype("float32")

index = faiss.read_index(str(ART / "faiss.index"))
meta  = pd.read_csv(ART / "meta.csv")

scores, ids = index.search(feat.reshape(1,-1), 1)
row = meta.iloc[int(ids[0][0])]
score = float(scores[0][0])

print("Most similar face:")
print("File name :", row["filename"])
print("Similarity:", round(score, 4))
print("Path      :", row["path"])
