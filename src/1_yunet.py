import cv2
import numpy as np
import time
from pathlib import Path


YUNET_MODEL = "../models/yunet/face_detection_yunet_2023mar.onnx"
SFACE_MODEL = "../models/sfacenet/face_recognition_sface_2021dec.onnx"
REF_IMG     = "../data/reference/me.jpg"

def l2norm(x: np.ndarray, eps=1e-12):
    return x / (np.linalg.norm(x) + eps)

def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    a = l2norm(a.astype(np.float32).reshape(-1))
    b = l2norm(b.astype(np.float32).reshape(-1))
    return float(np.dot(a, b))

def get_best_face(dets):
    if dets is None or len(dets) == 0:
        return None
    return max(dets, key=lambda d: d[4])

def main(cam_id=0, threshold=0.38):
    if not Path(YUNET_MODEL).exists():
        raise FileNotFoundError(f"YuNet model not found: {YUNET_MODEL}")
    if not Path(SFACE_MODEL).exists():
        raise FileNotFoundError(f"SFace model not found: {SFACE_MODEL}")
    if not Path(REF_IMG).exists():
        raise FileNotFoundError(f"Reference image not found: {REF_IMG}")

    detector = cv2.FaceDetectorYN.create(
        YUNET_MODEL,
        "",
        (320, 320),
        0.9,
        0.3,
        5000
    )
    recognizer = cv2.FaceRecognizerSF.create(SFACE_MODEL, "")

    ref_bgr = cv2.imread(REF_IMG)
    if ref_bgr is None:
        raise RuntimeError("Failed to read reference image")

    ref_h, ref_w = ref_bgr.shape[:2]
    detector.setInputSize((ref_w, ref_h))
    _, ref_dets = detector.detect(ref_bgr)

    ref_det = get_best_face(ref_dets)
    if ref_det is None:
        raise RuntimeError("No face found on reference image")

    ref_aligned = recognizer.alignCrop(ref_bgr, ref_det)
    ref_feat = recognizer.feature(ref_aligned)

    cap = cv2.VideoCapture(cam_id)
    if not cap.isOpened():
        raise RuntimeError("Cannot open webcam")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    prev_time = time.time()
    fps = 0.0

    print("[INFO] Webcam started. Press 'q' or ESC to exit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        H, W = frame.shape[:2]
        detector.setInputSize((W, H))
        _, dets = detector.detect(frame)
        
        now = time.time()
        dt = now - prev_time
        prev_time = now
        fps = 0.9 * fps + 0.1 * (1.0 / max(dt, 1e-6))

        if dets is not None:
            for d in dets:
                conf = float(d[4])
                if conf < 0.9:
                    continue

                aligned = recognizer.alignCrop(frame, d)
                feat = recognizer.feature(aligned)

                sim = cosine_sim(ref_feat, feat)
                is_me = sim >= threshold

                x, y, w, h = d[:4].astype(int)
                color = (0, 255, 0) if is_me else (0, 0, 255)

                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                label = f"{'ME' if is_me else 'OTHER'}  sim={sim:.3f}"
                cv2.putText(frame, label, (x, max(0, y - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        cv2.imshow("Face Verification (YuNet + SFace)", frame)

        key = cv2.waitKey(1)
        if key == 27 or key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main(cam_id=0, threshold=0.38)
