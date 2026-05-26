#!/usr/bin/env python3
"""
visualize_tracking.py
=====================
가장 안정적인 정상 면접 영상에 눈 추적 + 고개 탐지 결과를 오버레이하여
확인용 영상을 생성한다.

출력: data/viz/<video_id>_tracked.mp4

오버레이 항목
-------------
  - 얼굴 bbox (연두색)
  - Head Pose 3D 축 화살표  (X=빨강, Y=초록, Z=파랑)
  - 홍채 원 + 눈 윤곽 (청록/노랑)
  - 시선 방향 선
  - 수치 패널 (yaw / pitch / roll / eye_x_bias / eye_y_bias / eye_open)
  - 정상 범위 바 게이지
  - 탐지 여부 (얼굴 미검출 시 경고)
"""

import warnings, os
warnings.filterwarnings("ignore")
os.environ.setdefault("GLOG_minloglevel", "3")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

from pathlib import Path
import cv2
import numpy as np
import mediapipe as mp
from insightface.app import FaceAnalysis

# ── 공통 로직 재사용 ──────────────────────────────────────────────────────────
from extract_features import (
    estimate_head_pose,
    compute_eye_features,
    _MODEL_POINTS,
)

def _camera_matrix(w: int, h: int) -> np.ndarray:
    f = float(w)
    return np.array([[f, 0, w/2.0],
                     [0, f, h/2.0],
                     [0, 0,  1.0 ]], dtype=np.float64)

# ══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════════════

TARGET_VIDEO = Path("data/raw_videos/S-e9-bW4seo.004.mp4")
OUTPUT_DIR   = Path("data/viz")
OUTPUT_FPS   = 15          # 출력 영상 FPS (원본 FPS 그대로 재생)

# 정상 범위 (compute_normal_ranges.py 결과 기준)
NORMAL = {
    "head_yaw":    (-27.84, 39.83),
    "head_pitch":  (-26.92, 26.96),
    "head_roll":   (-14.02, 16.14),
    "eye_x_ratio": ( 0.441,  0.570),
    "eye_y_ratio": ( 0.221,  0.801),
}

# MediaPipe iris/eye indices
LEFT_IRIS  = [468, 469, 470, 471, 472]
RIGHT_IRIS = [473, 474, 475, 476, 477]
LEFT_EYE_CONTOUR  = [33,246,161,160,159,158,157,173,133,155,154,153,145,144,163,7]
RIGHT_EYE_CONTOUR = [362,398,384,385,386,387,388,466,263,249,390,373,374,380,381,382]

# 색상 (BGR)
C_GREEN  = (80, 220, 80)
C_RED    = (60, 60, 220)
C_BLUE   = (220, 100, 40)
C_YELLOW = (30, 220, 220)
C_CYAN   = (220, 200, 30)
C_WHITE  = (240, 240, 240)
C_GRAY   = (140, 140, 140)
C_DARK   = (30,  30,  30)
C_ORANGE = (30, 160, 255)


# ══════════════════════════════════════════════════════════════════════════════
# 시각화 유틸
# ══════════════════════════════════════════════════════════════════════════════

def draw_head_pose_axes(frame, rvec, tvec, cam_mat, dist, nose_2d, scale=80):
    """
    3D 좌표축을 얼굴 코 위치에 투영하여 화살표로 그린다.
    X=빨강(우), Y=초록(상), Z=파랑(전방→화면 밖)
    """
    axis_3d = np.array([
        [scale,     0,     0],   # X
        [    0, -scale,    0],   # Y (이미지 y축 반전)
        [    0,     0, scale],   # Z
    ], dtype=np.float64)

    pts, _ = cv2.projectPoints(axis_3d, rvec, tvec, cam_mat, dist)
    pts    = pts.reshape(-1, 2).astype(int)
    origin = tuple(nose_2d.astype(int))

    colors = [C_RED, C_GREEN, C_BLUE]
    labels = ["X", "Y", "Z"]
    for pt, color, lbl in zip(pts, colors, labels):
        end = tuple(pt)
        cv2.arrowedLine(frame, origin, end, color, 3, tipLength=0.25)
        cv2.putText(frame, lbl, (end[0]+4, end[1]-4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)


def draw_iris(frame, landmarks, indices, color, w, h, radius_override=None):
    """홍채 랜드마크를 원으로 그린다."""
    pts = np.array([[landmarks[i].x * w, landmarks[i].y * h]
                    for i in indices], dtype=np.float32)
    center = pts.mean(axis=0).astype(int)
    if radius_override:
        r = radius_override
    else:
        r = max(3, int(np.linalg.norm(pts[0] - pts[2]) / 2))
    cv2.circle(frame, tuple(center), r, color, 2, cv2.LINE_AA)
    cv2.circle(frame, tuple(center), 2, color, -1, cv2.LINE_AA)
    return center


def draw_eye_contour(frame, landmarks, indices, color, w, h):
    """눈 윤곽 랜드마크를 연결해서 그린다."""
    pts = np.array([[int(landmarks[i].x * w), int(landmarks[i].y * h)]
                    for i in indices], dtype=np.int32)
    cv2.polylines(frame, [pts], isClosed=True, color=color,
                  thickness=1, lineType=cv2.LINE_AA)


def draw_gaze_line(frame, left_center, right_center,
                   eye_x_bias, eye_y_bias, length=60):
    """양쪽 홍채 중심 평균에서 시선 방향 선을 그린다."""
    cx = int((left_center[0] + right_center[0]) / 2)
    cy = int((left_center[1] + right_center[1]) / 2)
    dx = int(eye_x_bias * length * 3)
    dy = int(eye_y_bias * length * 2)
    end = (cx + dx, cy + dy)
    cv2.arrowedLine(frame, (cx, cy), end, C_YELLOW, 2, tipLength=0.35, line_type=cv2.LINE_AA)


def draw_gauge(frame, x, y, w, h, value, lo, hi, label, color, unit="°"):
    """수평 게이지 바를 그린다."""
    # 배경
    cv2.rectangle(frame, (x, y), (x + w, y + h), (50, 50, 50), -1)
    cv2.rectangle(frame, (x, y), (x + w, y + h), C_GRAY, 1)

    # 정상 범위 내 채우기
    span = hi - lo
    if span < 1e-6:
        return
    ratio = np.clip((value - lo) / span, 0.0, 1.0)
    fill_w = int(w * ratio)
    bar_color = C_GREEN if lo < value < hi else C_RED
    cv2.rectangle(frame, (x, y + 1), (x + fill_w, y + h - 1), bar_color, -1)

    # 중앙선(정상 범위 중간)
    mid_ratio = np.clip((0 - lo) / span, 0.0, 1.0)  # 0도 기준선
    mid_x = x + int(w * mid_ratio)
    cv2.line(frame, (mid_x, y), (mid_x, y + h), C_WHITE, 1)

    # 레이블 + 수치
    txt = f"{label}: {value:+.1f}{unit}"
    cv2.putText(frame, txt, (x + w + 6, y + h - 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1, cv2.LINE_AA)


def draw_panel(frame, data: dict, panel_x=10, panel_y=10):
    """
    왼쪽 상단에 반투명 패널을 그리고 수치 및 게이지를 표시한다.
    data: {label: (value, lo, hi, unit, color)}
    """
    pw, ph = 320, len(data) * 28 + 20
    overlay = frame.copy()
    cv2.rectangle(overlay, (panel_x, panel_y),
                  (panel_x + pw, panel_y + ph), C_DARK, -1)
    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)
    cv2.rectangle(frame, (panel_x, panel_y),
                  (panel_x + pw, panel_y + ph), C_GRAY, 1)

    cv2.putText(frame, "HEAD & EYE TRACKING",
                (panel_x + 8, panel_y + 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, C_WHITE, 1, cv2.LINE_AA)

    gy = panel_y + 22
    for label, (value, lo, hi, unit, color) in data.items():
        draw_gauge(frame, panel_x + 8, gy, 160, 14,
                   value, lo, hi, label, color, unit)
        gy += 28


def draw_timestamp(frame, frame_idx, fps, w):
    t = frame_idx / fps
    txt = f"frame {frame_idx:04d}  |  {t:.2f}s"
    (tw, _), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
    cv2.putText(frame, txt,
                (w - tw - 10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, C_GRAY, 1, cv2.LINE_AA)


def draw_no_face(frame):
    h, w = frame.shape[:2]
    cv2.putText(frame, "NO FACE DETECTED",
                (w // 2 - 120, h // 2),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, C_RED, 2, cv2.LINE_AA)


# ══════════════════════════════════════════════════════════════════════════════
# 메인 처리
# ══════════════════════════════════════════════════════════════════════════════

def process_frame(frame, face_app, mesh, frame_idx):
    """단일 프레임에 모든 오버레이를 그린다."""
    h, w = frame.shape[:2]
    rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    dist  = np.zeros((4, 1), dtype=np.float64)
    cam   = _camera_matrix(w, h)

    # ── InsightFace ────────────────────────────────────────────────────────
    faces = []
    try:
        faces = face_app.get(rgb)
    except Exception:
        pass

    panel_data = {}

    if faces:
        face = max(faces,
                   key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]))
        x1, y1, x2, y2 = face.bbox.astype(int)

        # 얼굴 bbox
        cv2.rectangle(frame, (x1, y1), (x2, y2), C_GREEN, 2, cv2.LINE_AA)
        conf_txt = f"conf {face.det_score:.2f}"
        cv2.putText(frame, conf_txt, (x1, y1 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, C_GREEN, 1, cv2.LINE_AA)

        # Head pose
        kps = getattr(face, "kps", None)
        if kps is not None and kps.shape == (5, 2):
            pitch, yaw, roll = estimate_head_pose(kps, w, h)

            if not np.isnan(yaw):
                # 코 좌표 (kps[0])
                nose_2d = kps[2].astype(np.float64)

                # rvec 다시 계산 (시각화용 tvec 필요)
                image_pts = np.array([kps[2], kps[0], kps[1], kps[3], kps[4]],
                                     dtype=np.float64)
                _, rvec, tvec = cv2.solvePnP(
                    _MODEL_POINTS, image_pts, cam, dist,
                    flags=cv2.SOLVEPNP_EPNP
                )
                draw_head_pose_axes(frame, rvec, tvec, cam, dist, nose_2d)

                panel_data["Yaw"]   = (yaw,   *NORMAL["head_yaw"],   "°", C_RED)
                panel_data["Pitch"] = (pitch, *NORMAL["head_pitch"],  "°", C_ORANGE)
                panel_data["Roll"]  = (roll,  *NORMAL["head_roll"],   "°", C_BLUE)

    else:
        draw_no_face(frame)

    # ── MediaPipe FaceMesh ─────────────────────────────────────────────────
    try:
        mp_result = mesh.process(rgb)
    except Exception:
        mp_result = None

    if mp_result and mp_result.multi_face_landmarks:
        lm = mp_result.multi_face_landmarks[0].landmark

        # 눈 윤곽
        draw_eye_contour(frame, lm, LEFT_EYE_CONTOUR,  C_CYAN,   w, h)
        draw_eye_contour(frame, lm, RIGHT_EYE_CONTOUR, C_CYAN,   w, h)

        # 홍채
        lc = draw_iris(frame, lm, LEFT_IRIS,  C_YELLOW, w, h)
        rc = draw_iris(frame, lm, RIGHT_IRIS, C_YELLOW, w, h)

        # eye features
        ef = compute_eye_features(lm, w, h)
        if ef:
            draw_gaze_line(frame, lc, rc,
                           ef["eye_x_bias"], ef["eye_y_bias"])

            panel_data["EyeX bias"] = (
                ef["eye_x_bias"], -0.5, 0.5, "", C_CYAN)
            panel_data["EyeY bias"] = (
                ef["eye_y_bias"], -0.5, 0.5, "", C_CYAN)
            panel_data["Eye open"]  = (
                ef["eye_open"],  0.0,  0.5, "", C_GREEN)

    # ── 패널 ──────────────────────────────────────────────────────────────
    if panel_data:
        draw_panel(frame, panel_data)

    draw_timestamp(frame, frame_idx, OUTPUT_FPS, w)
    return frame


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not TARGET_VIDEO.exists():
        raise FileNotFoundError(f"영상 없음: {TARGET_VIDEO}")

    print("Loading InsightFace buffalo_l ...")
    face_app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
    face_app.prepare(ctx_id=0, det_size=(640, 640))

    print("Loading MediaPipe FaceMesh ...")
    mp_face_mesh = mp.solutions.face_mesh
    mesh = mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    cap = cv2.VideoCapture(str(TARGET_VIDEO))
    native_fps  = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out_path = OUTPUT_DIR / f"{TARGET_VIDEO.stem}_tracked.mp4"
    fourcc   = cv2.VideoWriter_fourcc(*"mp4v")
    writer   = cv2.VideoWriter(str(out_path), fourcc,
                               native_fps, (orig_w, orig_h))

    print(f"Processing: {TARGET_VIDEO.name}  ({orig_w}×{orig_h}, {native_fps:.1f}fps, {total_frames}frames)")

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        annotated = process_frame(frame, face_app, mesh, frame_idx)
        writer.write(annotated)
        frame_idx += 1
        if frame_idx % 30 == 0:
            print(f"  {frame_idx}/{total_frames} frames ...", end="\r")

    cap.release()
    writer.release()
    mesh.close()

    print(f"\n완료 → {out_path.resolve()}")


if __name__ == "__main__":
    main()
