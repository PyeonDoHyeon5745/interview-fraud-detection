#!/usr/bin/env python3
"""
render_analysis_video.py  —  PPT 발표용 분석 시각화 영상
=========================================================
실시간 InsightFace + MediaPipe 탐지 결과를 오버레이한다.

사용법:
  python render_analysis_video.py --video data/test_videos/2\ 좌+우\ 반복.mov
"""
import argparse, warnings, os
warnings.filterwarnings("ignore")
os.environ.setdefault("GLOG_minloglevel", "3")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

from pathlib import Path
from collections import deque

import cv2
import numpy as np
import pandas as pd
import mediapipe as mp
from insightface.app import FaceAnalysis
from PIL import Image, ImageDraw, ImageFont

from extract_features import estimate_head_pose, compute_eye_features, _MODEL_POINTS

FONT_PATH  = "/Library/Fonts/AppleSDGothicNeo.ttc"
RENDER_DIR = Path("data/results/rendered")
FRAME_DIR  = Path("data/results/test")

THR_YAW_DELTA   = 27.919
THR_PITCH_DELTA = 25.354
THR_EYE_X_LO   = 0.440
THR_EYE_X_HI   = 0.570

GRAPH_SECS  = 8
PANEL_RATIO = 0.30   # 우측 패널 비율

LEFT_IRIS  = [468, 469, 470, 471, 472]
RIGHT_IRIS = [473, 474, 475, 476, 477]
LEFT_EYE   = [33,246,161,160,159,158,157,173,133,155,154,153,145,144,163,7]
RIGHT_EYE  = [362,398,384,385,386,387,388,466,263,249,390,373,374,380,381,382]

# ── 색상 팔레트 ────────────────────────────────────────────────────────────────
C_BG          = (10,  12,  20)
C_PANEL       = (14,  16,  26)
C_GRID        = (35,  38,  55)
C_BORDER      = (55,  60,  90)
C_NORMAL_ZONE = (20,  48,  20)
C_THR_LINE    = (60, 140,  60)

C_YAW_LINE    = (110, 170, 255)   # 파란 계열
C_PITCH_LINE  = (120, 230, 130)   # 초록 계열
C_EX_LINE     = (255, 200,  80)   # 노란 계열
C_OVER_LINE   = (230,  65,  65)   # 빨강

C_MARKER_OK   = (255, 240, 100)
C_MARKER_OVER = (255,  80,  80)

C_STATUS_OK   = (30, 180, 80)
C_STATUS_OVER = (220, 60, 60)

C_HEAD_X      = (60,  60, 220)   # X축 (빨강 계열 BGR)
C_HEAD_Y      = (50, 220,  50)   # Y축 (초록)
C_HEAD_Z      = (220, 100, 40)   # Z축 (파랑 계열)
C_IRIS        = (30, 220, 220)
C_EYE_CTR     = (220, 200, 30)


def _font(size: int):
    try:
        return ImageFont.truetype(FONT_PATH, size, index=0)
    except Exception:
        return ImageFont.load_default()


def _camera_matrix(w, h):
    f = float(w)
    return np.array([[f,0,w/2],[0,f,h/2],[0,0,1]], dtype=np.float64)


def _text_w(draw, text, font):
    bb = draw.textbbox((0,0), text, font=font)
    return bb[2] - bb[0]


# ── 좌측 오버레이 ──────────────────────────────────────────────────────────────

def draw_head_axes(frame, rvec, tvec, cam, dist, nose_pt, scale=80):
    axis = np.array([[scale,0,0],[0,-scale,0],[0,0,scale]], dtype=np.float64)
    pts, _ = cv2.projectPoints(axis, rvec, tvec, cam, dist)
    pts    = pts.reshape(-1,2).astype(int)
    origin = tuple(nose_pt.astype(int))
    for pt, color in zip(pts, [C_HEAD_X, C_HEAD_Y, C_HEAD_Z]):
        cv2.arrowedLine(frame, origin, tuple(pt), color, 3,
                        tipLength=0.28, line_type=cv2.LINE_AA)


def draw_iris_and_eye(frame, lm, w, h):
    def iris_center(idxs):
        pts = np.array([[lm[i].x*w, lm[i].y*h] for i in idxs], np.float32)
        return pts.mean(axis=0).astype(int)

    def eye_contour(idxs, color):
        pts = np.array([[int(lm[i].x*w), int(lm[i].y*h)] for i in idxs], np.int32)
        cv2.polylines(frame, [pts], True, color, 1, cv2.LINE_AA)

    lc = iris_center(LEFT_IRIS)
    rc = iris_center(RIGHT_IRIS)

    l_pts = np.array([[lm[i].x*w, lm[i].y*h] for i in LEFT_IRIS], np.float32)
    r_pts = np.array([[lm[i].x*w, lm[i].y*h] for i in RIGHT_IRIS], np.float32)
    lr = max(4, int(np.linalg.norm(l_pts[0]-l_pts[2])/2))
    rr = max(4, int(np.linalg.norm(r_pts[0]-r_pts[2])/2))

    eye_contour(LEFT_EYE,  C_EYE_CTR)
    eye_contour(RIGHT_EYE, C_EYE_CTR)
    cv2.circle(frame, tuple(lc), lr, C_IRIS, 2, cv2.LINE_AA)
    cv2.circle(frame, tuple(rc), rr, C_IRIS, 2, cv2.LINE_AA)
    cv2.circle(frame, tuple(lc), 3,  C_IRIS, -1)
    cv2.circle(frame, tuple(rc), 3,  C_IRIS, -1)

    return lc, rc


def draw_gaze(frame, lc, rc, ex_bias, ey_bias, length=60):
    cx = int((lc[0]+rc[0])/2)
    cy = int((lc[1]+rc[1])/2)
    dx = int(ex_bias * length * 3)
    dy = int(ey_bias * length * 2)
    cv2.arrowedLine(frame, (cx,cy), (cx+dx,cy+dy),
                    C_IRIS, 2, tipLength=0.35, line_type=cv2.LINE_AA)


# ── 우측 패널 ──────────────────────────────────────────────────────────────────

def _draw_graph_panel(draw: ImageDraw.ImageDraw,
                      buf, vmin, vmax, tlo, thi,
                      gx, gy, gw, gh,
                      label: str, cur_val: str,
                      line_color,
                      f_label, f_val):
    """단일 그래프 박스 그리기 (PIL)"""

    # 외곽 배경
    draw.rectangle([(gx, gy), (gx+gw, gy+gh)],
                   fill=C_BG, outline=C_BORDER, width=1)

    span = vmax - vmin
    if span < 1e-6:
        span = 1.0

    def py(v):
        return gy + gh - max(1, min(gh-1, int((v - vmin) / span * gh)))

    # 그리드 수평선 (5개)
    n_grid = 4
    for i in range(1, n_grid):
        yy = gy + int(gh * i / n_grid)
        draw.line([(gx+1, yy), (gx+gw-1, yy)], fill=C_GRID, width=1)

    # 정상 범위 음영
    y_tlo = py(tlo); y_thi = py(thi)
    top_y = min(y_tlo, y_thi); bot_y = max(y_tlo, y_thi)
    for yy in range(max(gy+1, top_y), min(gy+gh-1, bot_y)):
        draw.line([(gx+1, yy), (gx+gw-1, yy)], fill=C_NORMAL_ZONE)

    # 임계선
    draw.line([(gx, y_tlo), (gx+gw, y_tlo)], fill=C_THR_LINE, width=1)
    draw.line([(gx, y_thi), (gx+gw, y_thi)], fill=C_THR_LINE, width=1)

    # 데이터 라인 + 그라데이션 채우기
    n = len(buf)
    if n >= 2:
        pts = [(gx + int(i / (n-1) * (gw-1)), py(v)) for i, v in enumerate(buf)]

        # 채우기 (baseline = py(0) 또는 범위 중앙)
        base_y = py(0) if (vmin < 0 < vmax) else gy + gh
        for i in range(len(pts)-1):
            x0, y0 = pts[i]
            x1, y1 = pts[i+1]
            v = buf[i]
            over = not (tlo <= v <= thi)
            fc = (*C_OVER_LINE, 40) if over else (*line_color, 30)
            # 채우기 폴리곤
            fill_pts = [(x0, y0), (x1, y1),
                        (x1, min(base_y, gy+gh-1)), (x0, min(base_y, gy+gh-1))]
            draw.polygon(fill_pts, fill=fc)

        # 라인 (2px)
        for i in range(len(pts)-1):
            v = buf[i]
            over = not (tlo <= v <= thi)
            dc = C_OVER_LINE if over else line_color
            draw.line([pts[i], pts[i+1]], fill=dc, width=2)

        # 현재값 마커
        cur_y = pts[-1][1]
        cur_v = buf[-1]
        over  = not (tlo <= cur_v <= thi)
        mc = C_MARKER_OVER if over else C_MARKER_OK
        mx = pts[-1][0]
        draw.ellipse([(mx-4, cur_y-4), (mx+4, cur_y+4)], fill=mc)

    # 레이블 (좌상단)
    cur_v = buf[-1] if buf else (tlo+thi)/2
    over  = not (tlo <= cur_v <= thi)
    lc_text = (220, 90, 90) if over else (180, 210, 255)
    draw.text((gx+5, gy+4), label, font=f_label, fill=lc_text)

    # 현재 수치 (우상단)
    val_w = _text_w(draw, cur_val, f_val)
    vc    = C_MARKER_OVER if over else C_MARKER_OK
    draw.text((gx + gw - val_w - 5, gy+4), cur_val, font=f_val, fill=vc)


def draw_right_panel(frame_bgr, buf_yaw, buf_pitch, buf_ex, any_over):
    H, W = frame_bgr.shape[:2]
    pw   = int(W * PANEL_RATIO)
    px   = W - pw

    img  = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img)

    # 패널 배경 (반투명 오버레이)
    ov = img.copy()
    ImageDraw.Draw(ov).rectangle([(px, 0), (W, H)], fill=C_PANEL)
    img  = Image.blend(img, ov, alpha=0.88)
    draw = ImageDraw.Draw(img)

    # 좌측 구분선
    draw.line([(px, 0), (px, H)], fill=C_BORDER, width=2)

    pad      = max(8, int(H * 0.013))
    f_title  = _font(int(H * 0.032))
    f_label  = _font(max(12, int(H * 0.022)))
    f_val    = _font(max(12, int(H * 0.022)))
    f_status = _font(int(H * 0.030))

    # ── 헤더 ──────────────────────────────────────────────────────
    header_h = int(H * 0.055)
    draw.rectangle([(px, 0), (W, header_h)], fill=(18, 20, 35))
    draw.line([(px, header_h), (W, header_h)], fill=C_BORDER, width=1)
    title_x = px + pad
    draw.text((title_x, (header_h - int(H*0.032))//2),
              "실시간 분석 지표", font=f_title, fill=(180, 190, 255))

    # ── 그래프 3개 ─────────────────────────────────────────────────
    footer_h = int(H * 0.075)
    avail    = H - header_h - footer_h - pad * 4
    g_h      = avail // 3
    g_w      = pw - pad * 2
    g_x      = px + pad

    yaw_cur   = buf_yaw[-1]   if buf_yaw   else 0.0
    pitch_cur = buf_pitch[-1] if buf_pitch else 0.0
    ex_cur    = buf_ex[-1]    if buf_ex    else 0.5

    graphs = [
        (buf_yaw,
         -THR_YAW_DELTA*1.6,  THR_YAW_DELTA*1.6,
         -THR_YAW_DELTA,       THR_YAW_DELTA,
         "Yaw Δ",              f"{yaw_cur:+.1f}°",
         C_YAW_LINE),
        (buf_pitch,
         -THR_PITCH_DELTA*1.6, THR_PITCH_DELTA*1.6,
         -THR_PITCH_DELTA,     THR_PITCH_DELTA,
         "Pitch Δ",            f"{pitch_cur:+.1f}°",
         C_PITCH_LINE),
        (buf_ex,
         0.26, 0.74,
         THR_EYE_X_LO,        THR_EYE_X_HI,
         "Eye-X",              f"{ex_cur:.3f}",
         C_EX_LINE),
    ]

    gy = header_h + pad
    for (buf, vmin, vmax, tlo, thi, label, cur_val, lc) in graphs:
        _draw_graph_panel(draw, buf, vmin, vmax, tlo, thi,
                          g_x, gy, g_w, g_h,
                          label, cur_val, lc,
                          f_label, f_val)
        gy += g_h + pad

    # ── 하단 상태 바 ──────────────────────────────────────────────
    status_y = H - footer_h
    if any_over:
        bg = (70, 12, 12)
        sc = C_STATUS_OVER
        st = "⚠  기준 초과!"
    else:
        bg = (10, 45, 20)
        sc = C_STATUS_OK
        st = "✓  정상 범위"

    draw.rectangle([(px, status_y), (W, H)], fill=bg)
    draw.line([(px, status_y), (W, status_y)], fill=C_BORDER, width=1)

    st_w = _text_w(draw, st, f_status)
    st_x = px + (pw - st_w) // 2
    st_y = status_y + (footer_h - int(H*0.030)) // 2
    # 그림자
    draw.text((st_x+1, st_y+1), st, font=f_status, fill=(0,0,0))
    draw.text((st_x,   st_y),   st, font=f_status, fill=sc)

    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


# ── 상단 HUD 바 ────────────────────────────────────────────────────────────────

def draw_hud(frame, t_sec, video_id, any_over):
    """상단 반투명 HUD: 영상명 + 타임스탬프 + 상태"""
    H, W = frame.shape[:2]
    pw = int(W * PANEL_RATIO)

    img  = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img)

    bar_h  = int(H * 0.052)
    ov     = img.copy()
    ImageDraw.Draw(ov).rectangle([(0,0),(W - pw, bar_h)], fill=(8,10,18))
    img    = Image.blend(img, ov, alpha=0.72)
    draw   = ImageDraw.Draw(img)
    draw.line([(0, bar_h), (W - pw, bar_h)], fill=C_BORDER, width=1)

    f_hud  = _font(int(H * 0.026))
    f_time = _font(int(H * 0.028))

    pad = int(H * 0.012)

    # 영상 이름
    draw.text((pad*2, (bar_h - int(H*0.026))//2),
              video_id, font=f_hud, fill=(190,195,230))

    # 타임스탬프 (우측 정렬)
    t_str = f"{int(t_sec)//60:02d}:{t_sec%60:05.2f}"
    tw    = _text_w(draw, t_str, f_time)
    draw.text((W - pw - tw - pad*2, (bar_h - int(H*0.028))//2),
              t_str, font=f_time, fill=(200, 210, 255))

    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


# ── 메인 렌더 ──────────────────────────────────────────────────────────────────

def render(video_path: Path):
    video_id = video_path.stem

    print("  InsightFace 로딩...")
    face_app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
    face_app.prepare(ctx_id=0, det_size=(640,640))

    print("  MediaPipe FaceMesh 로딩...")
    mesh = mp.solutions.face_mesh.FaceMesh(
        static_image_mode=False, max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5, min_tracking_confidence=0.5,
    )

    frame_csv = FRAME_DIR / f"{video_id}.csv"
    if frame_csv.exists():
        df = pd.read_csv(frame_csv)
        med_yaw   = float(df["head_yaw"].median())
        med_pitch = float(df["head_pitch"].median())
    else:
        med_yaw = med_pitch = 0.0
    print(f"  median — yaw:{med_yaw:.1f}°  pitch:{med_pitch:.1f}°")

    cap  = cv2.VideoCapture(str(video_path))
    fps  = cap.get(cv2.CAP_PROP_FPS) or 30.0
    W    = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H    = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    dist = np.zeros((4,1), dtype=np.float64)
    cam  = _camera_matrix(W, H)

    buf_n     = int(GRAPH_SECS * fps)
    buf_yaw   = deque([0.0]*buf_n, maxlen=buf_n)
    buf_pitch = deque([0.0]*buf_n, maxlen=buf_n)
    buf_ex    = deque([0.5]*buf_n, maxlen=buf_n)

    RENDER_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RENDER_DIR / f"{video_id}_analysis.mp4"
    out = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"avc1"), fps, (W, H))

    fidx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        yaw = pitch = 0.0
        ex  = 0.5
        ey_bias = 0.0
        any_over = False

        # InsightFace
        try:
            faces = face_app.get(rgb)
        except Exception:
            faces = []

        if faces:
            face = max(faces, key=lambda f:(f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]))
            x1,y1,x2,y2 = face.bbox.astype(int)

            kps = getattr(face, "kps", None)
            if kps is not None and kps.shape == (5,2):
                pitch, yaw, roll = estimate_head_pose(kps, W, H)
                if not np.isnan(yaw):
                    d_yaw   = yaw - med_yaw
                    d_pitch = pitch - med_pitch
                    yaw_over = abs(d_yaw)   > THR_YAW_DELTA
                    pit_over = d_pitch       > THR_PITCH_DELTA

                    box_c = (60,60,220) if (yaw_over or pit_over) else (50,220,50)
                    cv2.rectangle(frame, (x1,y1), (x2,y2), box_c, 3, cv2.LINE_AA)

                    image_pts = np.array([kps[2],kps[0],kps[1],kps[3],kps[4]], np.float64)
                    _, rvec, tvec = cv2.solvePnP(
                        _MODEL_POINTS, image_pts, cam, dist, flags=cv2.SOLVEPNP_EPNP)
                    draw_head_axes(frame, rvec, tvec, cam, dist, kps[2])

                    buf_yaw.append(d_yaw)
                    buf_pitch.append(d_pitch)
                    any_over = yaw_over or pit_over

        # MediaPipe
        mp_res = None
        try:
            mp_res = mesh.process(rgb)
        except Exception:
            pass

        if mp_res and mp_res.multi_face_landmarks:
            lm = mp_res.multi_face_landmarks[0].landmark
            lc_pt, rc_pt = draw_iris_and_eye(frame, lm, W, H)
            ef = compute_eye_features(lm, W, H)
            if ef:
                ex      = ef["eye_x_ratio"]
                ey_bias = ef["eye_y_bias"]
                draw_gaze(frame, lc_pt, rc_pt, ef["eye_x_bias"], ey_bias)
                if not (THR_EYE_X_LO <= ex <= THR_EYE_X_HI):
                    any_over = True

        buf_ex.append(ex)

        # 우측 패널
        frame = draw_right_panel(frame, buf_yaw, buf_pitch, buf_ex, any_over)

        # 상단 HUD
        t = fidx / fps
        frame = draw_hud(frame, t, video_id, any_over)

        out.write(frame)
        fidx += 1
        if fidx % 60 == 0:
            print(f"  {fidx} frames ({t:.1f}s)...", end="\r")

    cap.release()
    out.release()
    mesh.close()
    print(f"\n  [저장] {out_path.name}  ({fidx} frames)")
    print(f"  경로: {out_path.resolve()}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    args = parser.parse_args()
    print(f"\n분석 시각화 영상 생성: {args.video}")
    render(Path(args.video))


if __name__ == "__main__":
    main()
