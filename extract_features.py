#!/usr/bin/env python3
"""
Batch extraction of head-pose and eye features from interview videos.
Used to build normal-behaviour training data for anomaly detection.

Input  : data/raw_videos/*.mp4  (and .avi, .mov)
Output : data/frame_csv/<video_id>.csv  (one CSV per video)

Sampling rate : SAMPLE_FPS (default 5 fps)

Algorithm mirrors head_eye_angle.py / head_pose.angle.py exactly:
  - Head pose : InsightFace buffalo_l  →  solvePnP (EPNP)  →  custom Euler decomp
  - Eye feat  : MediaPipe FaceMesh (refine_landmarks=True)  →  project_ratio
"""

import os
import csv
import warnings
import traceback
from pathlib import Path

import numpy as np
import cv2

warnings.filterwarnings("ignore")

os.environ.setdefault("GLOG_minloglevel", "3")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import mediapipe as mp
from insightface.app import FaceAnalysis

# ══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════════════

SAMPLE_FPS = 5
INPUT_DIR  = Path("data/raw_videos")
OUTPUT_DIR = Path("data/frame_csv")

CSV_COLUMNS = [
    "video_id", "frame_idx", "time_sec",
    "face_detected", "face_conf",
    "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2",
    "head_yaw", "head_pitch", "head_roll",
    "eye_x_ratio", "eye_y_ratio",
    "eye_x_bias", "eye_y_bias",
    "eye_open",
]

# ── 3-D model points for solvePnP (same as head_eye_angle.py) ─────────────────
# Input kps order from InsightFace: [left_eye, right_eye, nose, left_mouth, right_mouth]
# We reorder to: [nose, left_eye, right_eye, left_mouth, right_mouth]
_MODEL_POINTS = np.array([
    ( 0.0,   0.0,   0.0),   # nose tip
    (-30.0, -30.0, -30.0),  # left  eye
    ( 30.0, -30.0, -30.0),  # right eye
    (-25.0,  30.0, -20.0),  # left  mouth corner
    ( 25.0,  30.0, -20.0),  # right mouth corner
], dtype=np.float64)

# ── MediaPipe landmark indices (refine_landmarks=True → 478 pts) ──────────────
# Same constants as head_eye_angle.py
LEFT_EYE_LEFT_CORNER  = 33
LEFT_EYE_RIGHT_CORNER = 133
LEFT_EYE_TOP          = 159
LEFT_EYE_BOTTOM       = 145
LEFT_IRIS             = [468, 469, 470, 471, 472]

RIGHT_EYE_LEFT_CORNER  = 362
RIGHT_EYE_RIGHT_CORNER = 263
RIGHT_EYE_TOP          = 386
RIGHT_EYE_BOTTOM       = 374
RIGHT_IRIS             = [473, 474, 475, 476, 477]


# ══════════════════════════════════════════════════════════════════════════════
# HEAD-POSE  (head_eye_angle.py / head_pose.angle.py と同一ロジック)
# ══════════════════════════════════════════════════════════════════════════════

def rotation_matrix_to_euler_angles(R: np.ndarray):
    """
    Identical to head_eye_angle.py.
    Returns (pitch, yaw, roll) in degrees.
    """
    sy = np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
    singular = sy < 1e-6

    if not singular:
        x = np.arctan2( R[2, 1],  R[2, 2])   # pitch
        y = np.arctan2(-R[2, 0],  sy)         # yaw
        z = np.arctan2( R[1, 0],  R[0, 0])   # roll
    else:
        x = np.arctan2(-R[1, 2],  R[1, 1])
        y = np.arctan2(-R[2, 0],  sy)
        z = 0.0

    return np.degrees([x, y, z])   # pitch, yaw, roll


def estimate_head_pose(kps: np.ndarray, img_w: int, img_h: int):
    """
    kps : (5, 2) InsightFace landmarks [left_eye, right_eye, nose, left_mouth, right_mouth]
    Returns (pitch, yaw, roll) in degrees, or (nan, nan, nan).
    """
    if kps is None or kps.shape != (5, 2):
        return np.nan, np.nan, np.nan

    # Reorder to match _MODEL_POINTS: nose first (same as reference code)
    image_points = np.array([
        kps[2],  # nose
        kps[0],  # left  eye
        kps[1],  # right eye
        kps[3],  # left  mouth
        kps[4],  # right mouth
    ], dtype=np.float64)

    focal = float(img_w)
    camera_matrix = np.array([
        [focal, 0,     img_w / 2.0],
        [0,     focal, img_h / 2.0],
        [0,     0,     1.0         ],
    ], dtype=np.float64)
    dist_coeffs = np.zeros((4, 1), dtype=np.float64)

    ok, rvec, _ = cv2.solvePnP(
        _MODEL_POINTS, image_points,
        camera_matrix, dist_coeffs,
        flags=cv2.SOLVEPNP_EPNP,        # same flag as reference
    )
    if not ok:
        return np.nan, np.nan, np.nan

    R, _ = cv2.Rodrigues(rvec)
    pitch, yaw, roll = rotation_matrix_to_euler_angles(R)
    return float(pitch), float(yaw), float(roll)


# ══════════════════════════════════════════════════════════════════════════════
# EYE FEATURES  (head_eye_angle.py と同一ロジック)
# ══════════════════════════════════════════════════════════════════════════════

def lm_to_xy(face_landmarks, idx: int, w: int, h: int) -> np.ndarray:
    lm = face_landmarks[idx]
    return np.array([lm.x * w, lm.y * h], dtype=np.float32)


def get_points(face_landmarks, indices, w: int, h: int) -> np.ndarray:
    return np.array([lm_to_xy(face_landmarks, i, w, h) for i in indices],
                    dtype=np.float32)


def project_ratio(point: np.ndarray, origin: np.ndarray, direction: np.ndarray):
    """Scalar projection of (point - origin) onto direction, normalised."""
    denom = np.dot(direction, direction)
    if denom < 1e-6:
        return None
    return float(np.dot(point - origin, direction) / denom)


def compute_single_eye_features(face_landmarks, w: int, h: int,
                                 left_corner_idx, right_corner_idx,
                                 top_idx, bottom_idx, iris_indices):
    left_corner  = lm_to_xy(face_landmarks, left_corner_idx,  w, h)
    right_corner = lm_to_xy(face_landmarks, right_corner_idx, w, h)
    top_pt       = lm_to_xy(face_landmarks, top_idx,          w, h)
    bottom_pt    = lm_to_xy(face_landmarks, bottom_idx,       w, h)
    iris_pts     = get_points(face_landmarks, iris_indices,    w, h)

    iris_center    = np.mean(iris_pts, axis=0)
    horizontal_vec = right_corner - left_corner
    vertical_vec   = bottom_pt   - top_pt

    x_ratio  = project_ratio(iris_center, left_corner, horizontal_vec)
    y_ratio  = project_ratio(iris_center, top_pt,      vertical_vec)

    eye_width  = float(np.linalg.norm(horizontal_vec))
    eye_height = float(np.linalg.norm(vertical_vec))
    openness   = eye_height / eye_width if eye_width > 1e-6 else None

    return x_ratio, y_ratio, openness


def compute_eye_features(face_landmarks, img_w: int, img_h: int):
    """
    Requires 478 landmarks (refine_landmarks=True).
    Returns dict or None.
    bias = (ratio - 0.5) * 2  →  same convention as head_eye_angle.py.
    """
    if len(face_landmarks) < 478:
        return None

    lx, ly, lo = compute_single_eye_features(
        face_landmarks, img_w, img_h,
        LEFT_EYE_LEFT_CORNER, LEFT_EYE_RIGHT_CORNER,
        LEFT_EYE_TOP, LEFT_EYE_BOTTOM,
        LEFT_IRIS,
    )
    rx, ry, ro = compute_single_eye_features(
        face_landmarks, img_w, img_h,
        RIGHT_EYE_LEFT_CORNER, RIGHT_EYE_RIGHT_CORNER,
        RIGHT_EYE_TOP, RIGHT_EYE_BOTTOM,
        RIGHT_IRIS,
    )

    if lx is None or rx is None or ly is None or ry is None:
        return None

    avg_x = (lx + rx) / 2.0
    avg_y = (ly + ry) / 2.0
    avg_o = ((lo or 0.0) + (ro or 0.0)) / 2.0

    return {
        "eye_x_ratio": round(avg_x, 6),
        "eye_y_ratio": round(avg_y, 6),
        "eye_x_bias":  round((avg_x - 0.5) * 2.0, 6),   # ← head_eye_angle.py 방식
        "eye_y_bias":  round((avg_y - 0.5) * 2.0, 6),
        "eye_open":    round(avg_o, 6),
    }


# ══════════════════════════════════════════════════════════════════════════════
# PER-VIDEO PROCESSING
# ══════════════════════════════════════════════════════════════════════════════

def _empty_row(video_id, frame_idx, time_sec) -> dict:
    row = {col: "" for col in CSV_COLUMNS}
    row["video_id"]      = video_id
    row["frame_idx"]     = frame_idx
    row["time_sec"]      = round(time_sec, 4)
    row["face_detected"] = 0
    return row


def process_video(video_path: Path,
                  face_app: FaceAnalysis,
                  mesh,
                  output_dir: Path) -> None:
    video_id = video_path.stem
    out_path = output_dir / f"{video_id}.csv"

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"  [WARN] cannot open: {video_path.name}")
        return

    native_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_step = max(1, round(native_fps / SAMPLE_FPS))

    rows      = []
    frame_idx = 0

    while True:
        ret, bgr = cap.read()
        if not ret:
            break

        if frame_idx % frame_step == 0:
            time_sec      = frame_idx / native_fps
            img_h, img_w  = bgr.shape[:2]
            rgb           = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

            row = _empty_row(video_id, frame_idx, time_sec)

            # ── InsightFace: detection + 5-pt kps → head pose ──────────────
            try:
                faces = face_app.get(rgb)
            except Exception:
                faces = []

            if faces:
                face = max(
                    faces,
                    key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]),
                )
                x1, y1, x2, y2 = face.bbox.astype(int)
                row["face_detected"] = 1
                row["face_conf"]     = round(float(face.det_score), 6)
                row["bbox_x1"]       = int(x1)
                row["bbox_y1"]       = int(y1)
                row["bbox_x2"]       = int(x2)
                row["bbox_y2"]       = int(y2)

                kps = getattr(face, "kps", None)
                pitch, yaw, roll = estimate_head_pose(kps, img_w, img_h)
                if not np.isnan(yaw):
                    row["head_yaw"]   = round(yaw,   4)
                    row["head_pitch"] = round(pitch,  4)
                    row["head_roll"]  = round(roll,   4)

            # ── MediaPipe FaceMesh: iris + eye landmarks ───────────────────
            try:
                mp_result = mesh.process(rgb)
            except Exception:
                mp_result = None

            if mp_result and mp_result.multi_face_landmarks:
                lm = mp_result.multi_face_landmarks[0].landmark
                ef = compute_eye_features(lm, img_w, img_h)
                if ef:
                    row.update(ef)

            rows.append(row)

        frame_idx += 1

    cap.release()

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    print(f"  [OK]  {len(rows):3d} frames  →  {out_path.name}")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not INPUT_DIR.exists():
        raise FileNotFoundError(
            f"Input directory not found: {INPUT_DIR.resolve()}\n"
            "  → Create data/raw_videos/ and place the .mp4 files there."
        )

    # InsightFace (buffalo_l : same as reference)
    print("Loading InsightFace buffalo_l ...")
    face_app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
    face_app.prepare(ctx_id=0, det_size=(640, 640))

    # MediaPipe FaceMesh
    print("Loading MediaPipe FaceMesh ...")
    mp_face_mesh = mp.solutions.face_mesh
    mesh = mp_face_mesh.FaceMesh(
        static_image_mode=True,       # each frame is independent (batch mode)
        max_num_faces=1,
        refine_landmarks=True,        # enables iris landmarks 468-477
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    exts = ("*.mp4", "*.avi", "*.mov", "*.MP4", "*.AVI", "*.MOV")
    video_files = sorted(p for ext in exts for p in INPUT_DIR.glob(ext))
    print(f"\nFound {len(video_files)} video(s) in {INPUT_DIR}\n")

    if not video_files:
        print("Nothing to process. Exiting.")
        mesh.close()
        return

    for i, vp in enumerate(video_files, 1):
        out_path = OUTPUT_DIR / f"{vp.stem}.csv"
        if out_path.exists():
            print(f"[{i:3d}/{len(video_files)}]  SKIP (already done)  {vp.name}")
            continue
        print(f"[{i:3d}/{len(video_files)}]  {vp.name}")
        try:
            process_video(vp, face_app, mesh, OUTPUT_DIR)
        except Exception:
            print(f"  [ERR] {vp.name}\n{traceback.format_exc()}")

    mesh.close()
    print(f"\nAll done. CSVs → {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
