#!/usr/bin/env python3
"""
render_analysis_toss.py  —  Toss 디자인 시스템 적용 분석 영상
=============================================================
insane-design/examples/toss/design.md 토큰 기반

Dark mode mapping:
  page bg   #17171C   (Toss dark section)
  card bg   #2C2C35   (inverseGrey100)
  border    #4D4D59   (inverseGrey300)
  text1     #FFFFFF   (inverseGrey900)
  text2     #E4E4E5   (inverseGrey800)
  text3     #C3C3C6   (inverseGrey700)
  brand     #3182F6   (Toss Blue)
  ok        #02A262   (green600)
  alert     #E42939   (red600)
  warn      #DD7D02   (yellow900)
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

# ── 설정 ────────────────────────────────────────────────────────────────────────
FONT_PATH  = "/Library/Fonts/AppleSDGothicNeo.ttc"
RENDER_DIR = Path("data/results/rendered")
FRAME_DIR  = Path("data/results/test")

THR_YAW_DELTA   = 27.919
THR_PITCH_DELTA = 25.354
THR_EYE_X_LO   = 0.440
THR_EYE_X_HI   = 0.570

GRAPH_SECS  = 8
PANEL_RATIO = 0.30

LEFT_IRIS  = [468, 469, 470, 471, 472]
RIGHT_IRIS = [473, 474, 475, 476, 477]
LEFT_EYE   = [33,246,161,160,159,158,157,173,133,155,154,153,145,144,163,7]
RIGHT_EYE  = [362,398,384,385,386,387,388,466,263,249,390,373,374,380,381,382]

# ── Toss Dark 컬러 토큰 ─────────────────────────────────────────────────────────
# BGR 포맷 (OpenCV)
_T_PAGE_BGR    = ( 28,  23,  23)   # #17171C
_T_CARD_BGR    = ( 53,  44,  44)   # #2C2C35
_T_BORDER_BGR  = ( 89,  77,  77)   # #4D4D59
_T_TEXT1_BGR   = (255, 255, 255)   # #FFFFFF
_T_TEXT2_BGR   = (229, 228, 228)   # #E4E4E5
_T_TEXT3_BGR   = (198, 195, 195)   # #C3C3C6
_T_BRAND_BGR   = (246, 130,  49)   # #3182F6
_T_OK_BGR      = ( 98, 162,   2)   # #02A262
_T_ALERT_BGR   = ( 57,  41, 233)   # #E42939
_T_WARN_BGR    = (  2, 125, 221)   # #DD7D02

# PIL RGBA 포맷
_T_PAGE_RGBA   = (23,  23,  28,  255)
_T_CARD_RGBA   = (44,  44,  53,  255)
_T_CARD_A88    = (44,  44,  53,  224)
_T_BORDER_RGBA = (77,  77,  89,  255)
_T_BORDER_SOFT = (77,  77,  89,  100)
_T_TEXT1_RGBA  = (255, 255, 255, 255)
_T_TEXT2_RGBA  = (228, 228, 229, 220)
_T_TEXT3_RGBA  = (195, 195, 198, 180)
_T_BRAND_RGBA  = (49,  130, 246, 255)
_T_BRAND_DIM   = (49,  130, 246,  40)
_T_OK_RGBA     = (2,   162,  98, 255)
_T_OK_DIM      = (2,   162,  98,  40)
_T_ALERT_RGBA  = (233,  41,  57, 255)
_T_ALERT_DIM   = (233,  41,  57,  40)
_T_WARN_RGBA   = (221, 125,   2, 255)
_T_WARN_DIM    = (221, 125,   2,  40)

# 그래프 라인 색 (PIL RGBA)
_G_YAW   = (49,  130, 246, 255)   # Toss Blue
_G_PITCH = (2,   162,  98, 255)   # Toss Green
_G_EX    = (221, 125,   2, 255)   # Toss Yellow

# 헤드 축 (OpenCV BGR)
C_HEAD_X = (246, 130, 49)
C_HEAD_Y = (98,  162,  2)
C_HEAD_Z = (2,  125, 221)
C_IRIS   = (246, 130, 49)   # brand blue (BGR)
C_EYE    = (198, 195, 195)  # text3

# ── 폰트 캐시 ────────────────────────────────────────────────────────────────────
_font_cache = {}
def _font(size: int):
    if size not in _font_cache:
        try:
            _font_cache[size] = ImageFont.truetype(FONT_PATH, size, index=0)
        except Exception:
            _font_cache[size] = ImageFont.load_default()
    return _font_cache[size]

def _tw(draw, text, font):
    bb = draw.textbbox((0,0), text, font=font)
    return bb[2] - bb[0]

def _th(draw, text, font):
    bb = draw.textbbox((0,0), text, font=font)
    return bb[3] - bb[1]

def _camera_matrix(w, h):
    f = float(w)
    return np.array([[f,0,w/2],[0,f,h/2],[0,0,1]], dtype=np.float64)


# ── 얼굴 오버레이 ────────────────────────────────────────────────────────────────
def draw_head_axes(frame, rvec, tvec, cam, dist, nose_pt, scale=80):
    axis = np.array([[scale,0,0],[0,-scale,0],[0,0,scale]], dtype=np.float64)
    pts, _ = cv2.projectPoints(axis, rvec, tvec, cam, dist)
    pts    = pts.reshape(-1,2).astype(int)
    origin = tuple(nose_pt.astype(int))
    for pt, color in zip(pts, [C_HEAD_X, C_HEAD_Y, C_HEAD_Z]):
        cv2.arrowedLine(frame, origin, tuple(pt), color, 2,
                        tipLength=0.25, line_type=cv2.LINE_AA)


def draw_face_box(frame, x1, y1, x2, y2, over: bool):
    """Toss 스타일 — 단색 라운드 박스 (코너만 그리기)"""
    color = _T_ALERT_BGR if over else _T_BRAND_BGR
    r     = 12
    thick = 2

    # 전체 사각형 얇게
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1, cv2.LINE_AA)

    # 모서리 강조 (Toss 카드 스타일)
    l = 20
    for (ax, ay, dx, dy) in [
        (x1, y1,  1,  1), (x2, y1, -1,  1),
        (x1, y2,  1, -1), (x2, y2, -1, -1),
    ]:
        cv2.line(frame, (ax, ay), (ax + dx*l, ay), color, thick, cv2.LINE_AA)
        cv2.line(frame, (ax, ay), (ax, ay + dy*l), color, thick, cv2.LINE_AA)


def draw_iris_and_eye(frame, lm, w, h):
    def iris_center(idxs):
        pts = np.array([[lm[i].x*w, lm[i].y*h] for i in idxs], np.float32)
        return pts.mean(axis=0).astype(int)

    def eye_contour(idxs):
        pts = np.array([[int(lm[i].x*w), int(lm[i].y*h)] for i in idxs], np.int32)
        cv2.polylines(frame, [pts], True, C_EYE, 1, cv2.LINE_AA)

    lc = iris_center(LEFT_IRIS)
    rc = iris_center(RIGHT_IRIS)

    l_pts = np.array([[lm[i].x*w, lm[i].y*h] for i in LEFT_IRIS], np.float32)
    r_pts = np.array([[lm[i].x*w, lm[i].y*h] for i in RIGHT_IRIS], np.float32)
    lr    = max(4, int(np.linalg.norm(l_pts[0]-l_pts[2])/2))
    rr    = max(4, int(np.linalg.norm(r_pts[0]-r_pts[2])/2))

    eye_contour(LEFT_EYE)
    eye_contour(RIGHT_EYE)
    cv2.circle(frame, tuple(lc), lr, C_IRIS, 1, cv2.LINE_AA)
    cv2.circle(frame, tuple(rc), rr, C_IRIS, 1, cv2.LINE_AA)
    cv2.circle(frame, tuple(lc), 3,  C_IRIS, -1)
    cv2.circle(frame, tuple(rc), 3,  C_IRIS, -1)
    return lc, rc


def draw_gaze(frame, lc, rc, ex_bias, ey_bias, length=60):
    cx = int((lc[0]+rc[0])/2)
    cy = int((lc[1]+rc[1])/2)
    dx = int(ex_bias * length * 3)
    dy = int(ey_bias * length * 2)
    cv2.arrowedLine(frame, (cx,cy), (cx+dx,cy+dy),
                    C_IRIS, 2, tipLength=0.3, line_type=cv2.LINE_AA)


# ── Toss 스타일 그래프 패널 ──────────────────────────────────────────────────────
def _rr(draw, xy, radius, fill=None, outline=None, width=1):
    """PIL rounded_rectangle 래퍼"""
    draw.rounded_rectangle(xy, radius=radius, fill=fill, outline=outline, width=width)


def _draw_toss_graph(draw: ImageDraw.ImageDraw,
                     buf, vmin, vmax, tlo, thi,
                     gx, gy, gw, gh,
                     label: str, cur_val: str,
                     line_rgba, fill_dim, alert_dim,
                     f_label, f_big):
    span = vmax - vmin
    if span < 1e-6:
        span = 1.0

    # 카드 배경 (라운드 12px — Toss radius-md)
    _rr(draw, [(gx, gy), (gx+gw, gy+gh)], 12,
        fill=_T_CARD_RGBA, outline=_T_BORDER_RGBA, width=1)

    def py(v):
        return gy + gh - max(2, min(gh-2, int((v - vmin) / span * (gh-4)))) - 2

    # 정상 구간 음영 (매우 연하게)
    y_tlo = py(tlo); y_thi = py(thi)
    top_y  = min(y_tlo, y_thi); bot_y = max(y_tlo, y_thi)
    if bot_y > top_y:
        ov = Image.new("RGBA", draw.im.size if hasattr(draw, 'im') else (1,1), (0,0,0,0))
        od = ImageDraw.Draw(ov)
        _rr(od, [(gx+2, top_y), (gx+gw-2, bot_y)], 4,
            fill=(*line_rgba[:3], 18))
        # 이건 _draw_toss_graph 내부에서 합성 불가 → 단순 fill로 대체
        draw.rectangle([(gx+2, top_y), (gx+gw-2, bot_y)],
                       fill=(*line_rgba[:3], 18))

    # 임계선 (점선 스타일)
    dash = 4
    for x in range(gx+8, gx+gw-4, dash*2):
        draw.line([(x, y_tlo), (x+dash, y_tlo)],
                  fill=(*line_rgba[:3], 60), width=1)
        draw.line([(x, y_thi), (x+dash, y_thi)],
                  fill=(*line_rgba[:3], 60), width=1)

    n = len(buf)
    if n >= 2:
        pts = [(gx + 2 + int(i / (n-1) * (gw-4)), py(v))
               for i, v in enumerate(buf)]

        # 채우기 폴리곤
        base_y = py(0) if (vmin < 0 < vmax) else gy + gh - 2
        poly   = []
        for x, y in pts:
            poly.append((x, y))
        poly += [(pts[-1][0], base_y), (pts[0][0], base_y)]
        # 현재값 over 여부
        cur_over = not (tlo <= buf[-1] <= thi)
        fill_c   = (*_T_ALERT_RGBA[:3], 25) if cur_over else (*line_rgba[:3], 22)
        draw.polygon(poly, fill=fill_c)

        # 라인 — 나이별 투명도
        for i in range(len(pts)-1):
            age  = (i+1) / n
            a    = int(55 + age * 200)
            v    = buf[i]
            over = not (tlo <= v <= thi)
            c    = (*_T_ALERT_RGBA[:3], a) if over else (*line_rgba[:3], a)
            draw.line([pts[i], pts[i+1]], fill=c, width=2)

        # 현재값 마커 (Toss 뱃지 스타일 — 작은 원)
        mx, my = pts[-1]
        mc = _T_ALERT_RGBA if cur_over else line_rgba
        for r, a in [(6, 30), (4, 70)]:
            draw.ellipse([(mx-r, my-r),(mx+r, my+r)], fill=(*mc[:3], a))
        draw.ellipse([(mx-2, my-2),(mx+2, my+2)], fill=mc)

    # ── 텍스트 영역 ────────────────────────────────────────────────────────────
    cur_v = buf[-1] if buf else (tlo+thi)/2
    over  = not (tlo <= cur_v <= thi)

    # 레이블 (좌상단, text3)
    draw.text((gx+10, gy+8), label, font=f_label, fill=_T_TEXT3_RGBA)

    # 현재 수치 — Toss badge 스타일 (우상단)
    vc_bg   = (*_T_ALERT_RGBA[:3], 40) if over else (*line_rgba[:3], 40)
    vc_text = _T_ALERT_RGBA            if over else line_rgba
    vw      = _tw(draw, cur_val, f_big)
    vh      = _th(draw, cur_val, f_big)
    bx      = gx + gw - vw - 20
    by      = gy + 6
    pad     = 5
    _rr(draw, [(bx-pad, by-2), (bx+vw+pad, by+vh+2)], 6, fill=vc_bg)
    draw.text((bx, by), cur_val, font=f_big, fill=vc_text)


# ── 우측 패널 전체 ────────────────────────────────────────────────────────────────
def draw_right_panel(frame_bgr, buf_yaw, buf_pitch, buf_ex, any_over):
    H, W = frame_bgr.shape[:2]
    pw   = int(W * PANEL_RATIO)
    px   = W - pw

    img  = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)).convert("RGBA")
    draw = ImageDraw.Draw(img, "RGBA")

    # 패널 배경 (#17171C)
    draw.rectangle([(px, 0), (W, H)], fill=_T_PAGE_RGBA)

    # 구분선 (border + brand blue 1px 포인트)
    draw.line([(px, 0), (px, H)], fill=_T_BORDER_RGBA, width=1)
    draw.line([(px+1, 0), (px+1, H)], fill=(*_T_BRAND_RGBA[:3], 60), width=1)

    pad      = max(10, int(H * 0.014))
    f_title  = _font(int(H * 0.028))
    f_label  = _font(max(11, int(H * 0.019)))
    f_big    = _font(max(13, int(H * 0.026)))
    f_status = _font(int(H * 0.028))

    # ── 헤더 ─────────────────────────────────────────────────────────────────
    header_h = int(H * 0.058)
    # 헤더 카드 (#2C2C35)
    draw.rectangle([(px, 0), (W, header_h)], fill=_T_CARD_RGBA)
    draw.line([(px, header_h), (W, header_h)], fill=_T_BORDER_RGBA, width=1)

    # 브랜드 블루 강조점 (Toss 특유의 포인트 컬러)
    dot_cx = px + pad + 5
    dot_cy = header_h // 2
    draw.ellipse([(dot_cx-4, dot_cy-4),(dot_cx+4, dot_cy+4)],
                 fill=(*_T_BRAND_RGBA[:3], 255))

    ty = (header_h - int(H*0.028)) // 2
    draw.text((dot_cx + 12, ty), "실시간 분석 지표",
              font=f_title, fill=_T_TEXT1_RGBA)

    # ── 그래프 3개 ─────────────────────────────────────────────────────────────
    footer_h = int(H * 0.095)
    avail    = H - header_h - footer_h - pad * 5
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
         "Yaw Δ",  f"{yaw_cur:+.1f}°",
         _G_YAW,   None, None),
        (buf_pitch,
         -THR_PITCH_DELTA*1.6, THR_PITCH_DELTA*1.6,
         -THR_PITCH_DELTA,     THR_PITCH_DELTA,
         "Pitch Δ", f"{pitch_cur:+.1f}°",
         _G_PITCH,  None, None),
        (buf_ex,
         0.24, 0.76,
         THR_EYE_X_LO,        THR_EYE_X_HI,
         "Eye-X",  f"{ex_cur:.3f}",
         _G_EX,    None, None),
    ]

    gy_cur = header_h + pad
    for (buf, vmin, vmax, tlo, thi, label, cur_val, lc, fd, ad) in graphs:
        _draw_toss_graph(draw, buf, vmin, vmax, tlo, thi,
                         g_x, gy_cur, g_w, g_h,
                         label, cur_val, lc, fd, ad,
                         f_label, f_big)
        gy_cur += g_h + pad

    # ── 하단 상태 바 — Toss Badge 스타일 ─────────────────────────────────────
    status_y = H - footer_h
    if any_over:
        bar_bg   = (*_T_ALERT_RGBA[:3], 220)
        badge_bg = (*_T_ALERT_RGBA[:3], 40)
        badge_fg = _T_ALERT_RGBA
        st       = "⚠  기준 초과"
    else:
        bar_bg   = (*_T_PAGE_RGBA[:3], 220)
        badge_bg = (*_T_OK_RGBA[:3], 40)
        badge_fg = _T_OK_RGBA
        st       = "✓  정상 범위"

    draw.rectangle([(px, status_y), (W, H)], fill=_T_CARD_RGBA)
    draw.line([(px, status_y), (W, status_y)], fill=_T_BORDER_RGBA, width=1)

    # 상태 뱃지 (Toss pill badge — radius 20px)
    sw  = _tw(draw, st, f_status)
    sh  = _th(draw, st, f_status)
    bx0 = px + (pw - sw) // 2 - 14
    by0 = status_y + (footer_h - sh) // 2 - 5
    _rr(draw, [(bx0, by0), (bx0 + sw + 28, by0 + sh + 10)],
        radius=20, fill=badge_bg, outline=(*badge_fg[:3], 80), width=1)
    draw.text((bx0 + 14, by0 + 5), st, font=f_status, fill=badge_fg)

    result = cv2.cvtColor(np.array(img.convert("RGB")), cv2.COLOR_RGB2BGR)
    return result


# ── 상단 HUD ─────────────────────────────────────────────────────────────────────
def draw_hud(frame, t_sec, total_sec, video_id, any_over, event_times):
    H, W = frame.shape[:2]
    pw   = int(W * PANEL_RATIO)
    vw   = W - pw

    img  = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).convert("RGBA")
    draw = ImageDraw.Draw(img, "RGBA")

    bar_h  = int(H * 0.052)
    prog_h = int(H * 0.016)

    # HUD 배경 (#2C2C35 카드 스타일)
    draw.rectangle([(0, 0), (vw, bar_h + prog_h)], fill=(*_T_CARD_RGBA[:3], 210))
    draw.line([(0, bar_h + prog_h), (vw, bar_h + prog_h)],
              fill=_T_BORDER_RGBA, width=1)

    f_name = _font(int(H * 0.022))
    f_time = _font(int(H * 0.024))
    pad    = int(H * 0.012)
    ty     = (bar_h - int(H*0.022)) // 2

    # 영상 이름 — text2 (#E4E4E5)
    draw.text((pad*2, ty), video_id, font=f_name, fill=_T_TEXT2_RGBA)

    # 상태 인디케이터 (작은 원) — brand or alert
    ind_c = _T_ALERT_RGBA if any_over else _T_BRAND_RGBA
    ix    = vw - pad*2 - int(H*0.024) - 8
    iy    = bar_h // 2
    draw.ellipse([(ix-4, iy-4),(ix+4, iy+4)], fill=(*ind_c[:3], 255))

    # 타임스탬프 — text3
    t_str = f"{int(t_sec)//60:02d}:{t_sec%60:05.2f}"
    tw    = _tw(draw, t_str, f_time)
    draw.text((ix - tw - 8, (bar_h - int(H*0.024))//2),
              t_str, font=f_time, fill=_T_TEXT3_RGBA)

    # ── 진행 바 (Toss progress bar style) ─────────────────────────────────────
    py0 = bar_h
    py1 = bar_h + prog_h
    # 배경 — inverseGrey300
    draw.rectangle([(0, py0), (vw, py1)], fill=(*_T_BORDER_RGBA[:3], 180))

    if total_sec > 0:
        prog   = min(1.0, t_sec / total_sec)
        fill_w = int(vw * prog)
        if fill_w > 0:
            _rr(draw, [(0, py0+1), (fill_w, py1-1)],
                radius=prog_h//2, fill=(*_T_BRAND_RGBA[:3], 220))

        # 이벤트 마커 (빨간 틱)
        for et in event_times:
            ex_ = int(vw * et / total_sec)
            draw.line([(ex_, py0), (ex_, py1)],
                      fill=(*_T_ALERT_RGBA[:3], 200), width=2)

    result = cv2.cvtColor(np.array(img.convert("RGB")), cv2.COLOR_RGB2BGR)
    return result


# ── 메인 렌더 ─────────────────────────────────────────────────────────────────────
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
    event_times = []
    if frame_csv.exists():
        df = pd.read_csv(frame_csv)
        med_yaw   = float(df["head_yaw"].median())
        med_pitch = float(df["head_pitch"].median())
        if "timestamp" in df.columns and "event" in df.columns:
            evt_df = df[df["event"].notna() & (df["event"] != "normal")]
            event_times = evt_df["timestamp"].tolist()
    else:
        med_yaw = med_pitch = 0.0
    print(f"  median — yaw:{med_yaw:.1f}°  pitch:{med_pitch:.1f}°")

    cap          = cv2.VideoCapture(str(video_path))
    fps          = cap.get(cv2.CAP_PROP_FPS) or 30.0
    W            = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H            = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    total_sec    = total_frames / fps
    dist         = np.zeros((4,1), dtype=np.float64)
    cam          = _camera_matrix(W, H)

    buf_n     = int(GRAPH_SECS * fps)
    buf_yaw   = deque([0.0]*buf_n, maxlen=buf_n)
    buf_pitch = deque([0.0]*buf_n, maxlen=buf_n)
    buf_ex    = deque([0.5]*buf_n, maxlen=buf_n)

    RENDER_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RENDER_DIR / f"{video_id}_toss.mp4"
    out = cv2.VideoWriter(str(out_path),
                          cv2.VideoWriter_fourcc(*"avc1"),
                          fps, (W, H))

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
                    over     = yaw_over or pit_over

                    draw_face_box(frame, x1, y1, x2, y2, over)

                    image_pts = np.array([kps[2],kps[0],kps[1],kps[3],kps[4]], np.float64)
                    _, rvec, tvec = cv2.solvePnP(
                        _MODEL_POINTS, image_pts, cam, dist, flags=cv2.SOLVEPNP_EPNP)
                    draw_head_axes(frame, rvec, tvec, cam, dist, kps[2])

                    buf_yaw.append(d_yaw)
                    buf_pitch.append(d_pitch)
                    any_over = over

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

        frame = draw_right_panel(frame, buf_yaw, buf_pitch, buf_ex, any_over)
        t     = fidx / fps
        frame = draw_hud(frame, t, total_sec, video_id, any_over, event_times)

        out.write(frame)
        fidx += 1
        if fidx % 30 == 0:
            print(f"  {fidx}/{total_frames} ({t:.1f}s)...", end="\r")

    cap.release()
    out.release()
    mesh.close()
    print(f"\n  [완료] {out_path.name}  ({fidx} frames)")
    print(f"  경로: {out_path.resolve()}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    args = parser.parse_args()
    print(f"\nToss 디자인 분석 영상 생성: {args.video}")
    render(Path(args.video))


if __name__ == "__main__":
    main()
