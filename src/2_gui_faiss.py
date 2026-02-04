import streamlit as st
import cv2
import numpy as np
import pandas as pd
import faiss
from pathlib import Path
import tempfile

ART = Path("../data/artifacts")
YUNET = "../models/yunet/face_detection_yunet_2023mar.onnx"
SFACE = "../models/sfacenet/face_recognition_sface_2021dec.onnx"

def l2norm(x):
    return x / (np.linalg.norm(x) + 1e-12)

@st.cache_resource
def load_models():
    detector = cv2.FaceDetectorYN.create(YUNET, "", (320,320), 0.6, 0.3, 5000)
    recognizer = cv2.FaceRecognizerSF.create(SFACE, "")
    index = faiss.read_index(str(ART / "faiss.index"))
    meta = pd.read_csv(ART / "meta.csv")
    return detector, recognizer, index, meta

st.title("MIEM Lookalike Search (SFace + FAISS)")

detector, recognizer, index, meta = load_models()

file = st.file_uploader("Upload face image", type=["jpg", "png", "jpeg"])

if file:
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(file.read())
        img_path = tmp.name

    img = cv2.imread(img_path)
    h, w = img.shape[:2]
    detector.setInputSize((w, h))
    _, dets = detector.detect(img)

    if dets is None or len(dets) == 0:
        st.error("No face detected")
    else:
        d = max(dets, key=lambda x: x[4])
        face = recognizer.alignCrop(img, d)
        feat = l2norm(recognizer.feature(face).reshape(-1)).astype("float32")

        scores, ids = index.search(feat.reshape(1,-1), 1)
        row = meta.iloc[int(ids[0][0])]
        score = float(scores[0][0])

        st.image(file, caption="Query image", use_container_width=True)
        st.success(f"Most similar: {row['filename']}")
        st.write(f"Similarity: {score:.4f}")
        st.image(row["path"], caption=row["filename"], use_container_width=True)
