#!/usr/bin/env python3
"""
render_analysis_v2.py  —  이쁜 분석 시각화 영상 (v2)
=====================================================
사용법:
  python render_analysis_v2.py --video "data/test_videos/2 좌+우 반복.mov"
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
from PIL import Image, ImageDraw, ImageFont, ImageFilter

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
PANEL_RATIO = 0.32   # 우측 패널 비율

LEFT_IRIS  = [468, 469, 470, 471, 472]
RIGHT_IRIS = [473, 474, 475, 476, 477]
LEFT_EYE   = [33,246,161,160,159,158,157,173,133,155,154,153,145,144,163,7]
RIGHT_EYE  = [362,398,384,385,386,387,388,466,263,249,390,373,374,380,381,382]

# ── 컬러 팔레트 ─────────────────────────────────────────────────────────────────
C_BG          = (8,  10,  18)
C_PANEL_TOP   = (16, 18,  32)
C_PANEL_BOT   = (10, 12,  22)
C_GRID        = (28, 32,  50)
C_BORDER      = (48, 54,  85)
C_BORDER_GLOW = (80, 90, 140)
C_NORMAL_ZONE = (15, 45,  22)
C_THR_LINE    = (50, 130,  55)

C_YAW_LINE    = (100, 165, 255)
C_PITCH_LINE  = (100, 225, 120)
C_EX_LINE     = (255, 195,  70)
C_OVER_LINE   = (235,  60,  60)

C_MARKER_OK   = (255, 245, 100)
C_MARKER_OVER = (255,  80,  80)
C_MARKER_GLOW = (255, 140,  40)

C_STATUS_OK   = ( 40, 200,  90)
C_STATUS_OVER = (230,  55,  55)

C_HEAD_X = (60,  60, 220)
C_HEAD_Y = (50, 220,  50)
C_HEAD_Z = (220, 110, 40)
C_IRIS   = (30, 225, 230)
C_EYE    = (230, 205,  35)

C_HUD_BG   = (8, 10, 18)
C_HUD_TEXT = (185, 192, 235)
C_PROG_BG  = (25, 28, 48)
C_PROG_FG  = (70, 120, 220)
C_PROG_EVT = (235, 65, 65)

# ── 폰트 ────────────────────────────────────────────────────────────────────────
_font_cache = {}
def _font(size: int):
    if size not in _font_cache:
        try:
            _font_cache[size] = ImageFont.truetype(FONT_PATH, size, index=0)
        except Exception:
            _font_cache[size] = ImageFont.load_default()
    return _font_cache[size]


def _camera_matrix(w, h):
    f = float(w)
    return np.array([[f,0,w/2],[0,f,h/2],[0,0,1]], dtype=np.float64)


def _text_w(draw, text, font):
    bb = draw.textbbox((0,0), text, font=font)
    return bb[2] - bb[0]

def _text_h(draw, text, font):
    bb = draw.textbbox((0,0), text, font=font)
    return bb[3] - bb[1]


# ── RGBA 오버레이 헬퍼 ──────────────────────────────────────────────────────────
def _alpha_rect(draw_img: Image.Image, xy, fill_rgba):
    ov = Image.new("RGBA", draw_img.size, (0,0,0,0))
    ImageDraw.Draw(ov).rectangle(xy, fill=fill_rgba)
    return Image.alpha_composite(draw_img.convert("RGBA"), ov).convert("RGB")


# ── 얼굴 오버레이 ───────────────────────────────────────────────────────────────
def draw_head_axes(frame, rvec, tvec, cam, dist, nose_pt, scale=80):
    axis = np.array([[scale,0,0],[0,-scale,0],[0,0,scale]], dtype=np.float64)
    pts, _ = cv2.projectPoints(axis, rvec, tvec, cam, dist)
    pts    = pts.reshape(-1,2).astype(int)
    origin = tuple(nose_pt.astype(int))
    for pt, color in zip(pts, [C_HEAD_X, C_HEAD_Y, C_HEAD_Z]):
        cv2.arrowedLine(frame, origin, tuple(pt), color, 3,
                        tipLength=0.28, line_type=cv2.LINE_AA)


def draw_face_box_glow(frame, x1, y1, x2, y2, over: bool):
    """글로우 효과가 있는 얼굴 박스"""
    if over:
        layers = [(10, (220, 40, 40, 10)), (6, (220, 40, 40, 25)),
                  (3, (220, 60, 60, 60)), (1, (220, 80, 80))]
    else:
        layers = [(8, (40, 200, 80, 12)), (4, (40, 200, 80, 30)),
                  (2, (60, 220, 80, 55)), (1, (60, 220, 80))]

    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).convert("RGBA")
    for expand, color in layers:
        ov = Image.new("RGBA", img.size, (0,0,0,0))
        d  = ImageDraw.Draw(ov)
        if len(color) == 4:
            d.rectangle([(x1-expand, y1-expand), (x2+expand, y2+expand)],
                        outline=(*color[:3], color[3]), width=2)
        else:
            d.rectangle([(x1-expand, y1-expand), (x2+expand, y2+expand)],
                        outline=(*color, 255), width=2)
        img = Image.alpha_composite(img, ov)

    result = cv2.cvtColor(np.array(img.convert("RGB")), cv2.COLOR_RGB2BGR)
    frame[:] = result[:]


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

    # 눈 외곽선 + 내부 반투명 채우기
    eye_contour(LEFT_EYE,  C_EYE)
    eye_contour(RIGHT_EYE, C_EYE)

    # 홍채 글로우
    cv2.circle(frame, tuple(lc), lr+3, (*C_IRIS[::-1][:2], C_IRIS[0]), 1, cv2.LINE_AA)
    cv2.circle(frame, tuple(rc), rr+3, (*C_IRIS[::-1][:2], C_IRIS[0]), 1, cv2.LINE_AA)
    cv2.circle(frame, tuple(lc), lr,   C_IRIS, 2, cv2.LINE_AA)
    cv2.circle(frame, tuple(rc), rr,   C_IRIS, 2, cv2.LINE_AA)
    cv2.circle(frame, tuple(lc), 3,    C_IRIS, -1)
    cv2.circle(frame, tuple(rc), 3,    C_IRIS, -1)
    return lc, rc


def draw_gaze(frame, lc, rc, ex_bias, ey_bias, length=60):
    cx = int((lc[0]+rc[0])/2)
    cy = int((lc[1]+rc[1])/2)
    dx = int(ex_bias * length * 3)
    dy = int(ey_bias * length * 2)
    # 글로우
    cv2.arrowedLine(frame, (cx,cy), (cx+dx,cy+dy),
                    (C_IRIS[0]//2, C_IRIS[1]//2, C_IRIS[2]//2),
                    5, tipLength=0.3, line_type=cv2.LINE_AA)
    cv2.arrowedLine(frame, (cx,cy), (cx+dx,cy+dy),
                    C_IRIS, 2, tipLength=0.35, line_type=cv2.LINE_AA)


# ── 그래프 패널 ─────────────────────────────────────────────────────────────────
def _smooth_pts(pts, win=5):
    """이동평균으로 점 스무딩"""
    if len(pts) < win*2:
        return pts
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    out = []
    for i in range(len(pts)):
        lo = max(0, i-win)
        hi = min(len(pts), i+win+1)
        out.append((xs[i], int(sum(ys[lo:hi])/(hi-lo))))
    return out


def _draw_graph_panel(draw: ImageDraw.ImageDraw,
                      buf, vmin, vmax, tlo, thi,
                      gx, gy, gw, gh,
                      label: str, cur_val: str,
                      line_color,
                      f_label, f_val, f_big):
    span = vmax - vmin
    if span < 1e-6:
        span = 1.0

    # 배경
    draw.rectangle([(gx, gy), (gx+gw, gy+gh)],
                   fill=C_BG, outline=C_BORDER, width=1)

    def py(v):
        return gy + gh - max(1, min(gh-1, int((v - vmin) / span * gh)))

    # 그리드 (수평 점선 느낌)
    for i in range(1, 5):
        yy = gy + int(gh * i / 4)
        for xx in range(gx+4, gx+gw-4, 6):
            draw.point((xx, yy), fill=C_GRID)

    # 정상 범위 음영 (그라디언트 느낌 - 여러 레이어)
    y_tlo = py(tlo); y_thi = py(thi)
    top_y  = min(y_tlo, y_thi); bot_y = max(y_tlo, y_thi)
    zone_h = max(1, bot_y - top_y)
    for yy in range(max(gy+1, top_y), min(gy+gh-1, bot_y)):
        ratio  = 1 - abs(yy - (top_y+bot_y)//2) / (zone_h/2+1)
        alpha  = int(18 + ratio * 20)
        draw.line([(gx+1, yy), (gx+gw-1, yy)],
                  fill=(C_NORMAL_ZONE[0], C_NORMAL_ZONE[1]+10, C_NORMAL_ZONE[2], alpha))

    # 임계선 (점선)
    for x in range(gx, gx+gw, 5):
        draw.line([(x, y_tlo), (min(x+3, gx+gw), y_tlo)], fill=C_THR_LINE, width=1)
        draw.line([(x, y_thi), (min(x+3, gx+gw), y_thi)], fill=C_THR_LINE, width=1)

    # 데이터 라인
    n = len(buf)
    if n >= 2:
        pts = [(gx + int(i / (n-1) * (gw-1)), py(v)) for i, v in enumerate(buf)]
        smooth = _smooth_pts(pts, win=3)

        base_y = py(0) if (vmin < 0 < vmax) else gy + gh

        # 채우기 - 나이에 따라 페이드
        for i in range(len(smooth)-1):
            x0, y0 = smooth[i]
            x1, y1 = smooth[i+1]
            v      = buf[i]
            over   = not (tlo <= v <= thi)
            age    = i / max(n-1, 1)     # 0=오래됨 1=최신
            alpha  = int(15 + age * 45)
            fc     = (*C_OVER_LINE, alpha) if over else (*line_color, alpha)
            fp     = [(x0, y0), (x1, y1),
                      (x1, min(base_y, gy+gh-1)), (x0, min(base_y, gy+gh-1))]
            draw.polygon(fp, fill=fc)

        # 라인 - 나이에 따라 페이드
        for i in range(len(smooth)-1):
            v    = buf[i]
            over = not (tlo <= v <= thi)
            age  = i / max(n-1, 1)
            a    = int(80 + age * 175)
            dc   = (*C_OVER_LINE, a) if over else (*line_color, a)
            draw.line([smooth[i], smooth[i+1]], fill=dc, width=2)

        # 현재값 마커 글로우
        if smooth:
            mx, my = smooth[-1]
            cur_v  = buf[-1]
            over   = not (tlo <= cur_v <= thi)
            mc     = C_MARKER_OVER if over else C_MARKER_OK
            mg     = C_OVER_LINE   if over else line_color
            for r, a in [(9, 30), (6, 60), (4, 120)]:
                draw.ellipse([(mx-r, my-r), (mx+r, my+r)], fill=(*mg, a))
            draw.ellipse([(mx-3, my-3), (mx+3, my+3)], fill=mc)

    # ── 레이블 + 큰 수치 ──────────────────────────────────────────────────────
    cur_v = buf[-1] if buf else (tlo+thi)/2
    over  = not (tlo <= cur_v <= thi)

    # 레이블 (좌상단, 작게)
    lc_text = (220, 90, 90) if over else (140, 165, 220)
    draw.text((gx+6, gy+4), label, font=f_label, fill=lc_text)

    # 큰 수치 (우상단)
    vc    = C_MARKER_OVER if over else C_MARKER_OK
    val_w = _text_w(draw, cur_val, f_big)
    # 수치 배경
    pad   = 4
    bx    = gx + gw - val_w - pad*2 - 4
    by    = gy + 3
    bh    = _text_h(draw, cur_val, f_big) + pad
    bw    = val_w + pad*2
    draw.rectangle([(bx, by), (bx+bw, by+bh)],
                   fill=(*(20,25,40), 200), outline=(*C_BORDER,), width=1)
    draw.text((bx+pad, by+pad//2), cur_val, font=f_big, fill=vc)


def draw_right_panel(frame_bgr, buf_yaw, buf_pitch, buf_ex, any_over):
    H, W = frame_bgr.shape[:2]
    pw   = int(W * PANEL_RATIO)
    px   = W - pw

    img  = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)).convert("RGBA")

    # 패널 배경 그라디언트 (상→하)
    grad = Image.new("RGBA", (pw, H), (0,0,0,0))
    gd   = ImageDraw.Draw(grad)
    for y in range(H):
        t   = y / H
        r   = int(C_PANEL_TOP[0]*(1-t) + C_PANEL_BOT[0]*t)
        g   = int(C_PANEL_TOP[1]*(1-t) + C_PANEL_BOT[1]*t)
        b   = int(C_PANEL_TOP[2]*(1-t) + C_PANEL_BOT[2]*t)
        gd.line([(0, y), (pw, y)], fill=(r,g,b, 225))
    img.paste(grad, (px, 0), grad)

    draw = ImageDraw.Draw(img, "RGBA")

    # 구분선 (글로우)
    for i, (offset, alpha) in enumerate([(2,15),(1,30),(0,180)]):
        draw.line([(px-offset, 0), (px-offset, H)],
                  fill=(*C_BORDER_GLOW, alpha), width=1)

    pad      = max(8, int(H * 0.013))
    f_title  = _font(int(H * 0.030))
    f_label  = _font(max(11, int(H * 0.020)))
    f_val    = _font(max(11, int(H * 0.020)))
    f_big    = _font(max(14, int(H * 0.028)))
    f_status = _font(int(H * 0.030))

    # ── 헤더 ─────────────────────────────────────────────────────────────────
    header_h = int(H * 0.060)
    draw.rectangle([(px, 0), (W, header_h)], fill=(12, 14, 28, 240))
    # 헤더 하단 강조선
    draw.line([(px, header_h), (W, header_h)], fill=(*C_BORDER_GLOW, 200), width=2)
    draw.line([(px, header_h+1), (W, header_h+1)], fill=(*C_BORDER_GLOW, 50), width=1)

    # 헤더 텍스트 + 작은 장식
    dot_x = px + pad + 5
    dot_y = header_h // 2
    for r, c in [(6,(70,130,220,40)),(4,(90,150,240,80)),(3,(120,180,255,200))]:
        draw.ellipse([(dot_x-r, dot_y-r),(dot_x+r, dot_y+r)], fill=c)
    draw.text((dot_x + 12, (header_h - int(H*0.030))//2),
              "실시간 분석 지표", font=f_title, fill=(185, 195, 255, 255))

    # ── 그래프 3개 ────────────────────────────────────────────────────────────
    footer_h = int(H * 0.090)
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
         "Yaw Δ",              f"{yaw_cur:+.1f}°",
         C_YAW_LINE),
        (buf_pitch,
         -THR_PITCH_DELTA*1.6, THR_PITCH_DELTA*1.6,
         -THR_PITCH_DELTA,     THR_PITCH_DELTA,
         "Pitch Δ",            f"{pitch_cur:+.1f}°",
         C_PITCH_LINE),
        (buf_ex,
         0.24, 0.76,
         THR_EYE_X_LO,        THR_EYE_X_HI,
         "Eye-X",              f"{ex_cur:.3f}",
         C_EX_LINE),
    ]

    gy = header_h + pad
    for (buf, vmin, vmax, tlo, thi, label, cur_val, lc) in graphs:
        _draw_graph_panel(draw, buf, vmin, vmax, tlo, thi,
                          g_x, gy, g_w, g_h,
                          label, cur_val, lc,
                          f_label, f_val, f_big)
        gy += g_h + pad

    # ── 하단 상태 바 ──────────────────────────────────────────────────────────
    status_y = H - footer_h
    if any_over:
        bg1  = (80, 12, 12, 230)
        bg2  = (55,  8,  8, 230)
        sc   = C_STATUS_OVER
        st   = "⚠  기준 초과"
        accent = (*C_OVER_LINE, 200)
    else:
        bg1  = (10, 55, 22, 230)
        bg2  = ( 8, 40, 18, 230)
        sc   = C_STATUS_OK
        st   = "✓  정상 범위"
        accent = (*C_STATUS_OK, 200)

    # 그라디언트 배경
    for y in range(status_y, H):
        t = (y - status_y) / max(1, footer_h)
        r = int(bg1[0]*(1-t) + bg2[0]*t)
        g = int(bg1[1]*(1-t) + bg2[1]*t)
        b = int(bg1[2]*(1-t) + bg2[2]*t)
        draw.line([(px, y), (W, y)], fill=(r,g,b, 230))

    draw.line([(px, status_y), (W, status_y)], fill=(*C_BORDER, 180), width=1)
    draw.line([(px, status_y+1), (W, status_y+1)], fill=accent, width=1)

    # 상태 텍스트 (그림자 + 본문)
    st_w = _text_w(draw, st, f_status)
    st_x = px + (pw - st_w) // 2
    st_y = status_y + (footer_h - int(H*0.030)) // 2
    draw.text((st_x+1, st_y+2), st, font=f_status, fill=(0,0,0,180))
    draw.text((st_x,   st_y),   st, font=f_status, fill=(*sc, 255))

    result = cv2.cvtColor(np.array(img.convert("RGB")), cv2.COLOR_RGB2BGR)
    return result


# ── 상단 HUD ────────────────────────────────────────────────────────────────────
def draw_hud(frame, t_sec, total_sec, video_id, any_over, event_times):
    H, W = frame.shape[:2]
    pw   = int(W * PANEL_RATIO)
    vid_w = W - pw

    img  = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).convert("RGBA")
    draw = ImageDraw.Draw(img, "RGBA")

    bar_h  = int(H * 0.055)
    prog_h = int(H * 0.018)
    total_h = bar_h + prog_h

    # HUD 배경 그라디언트
    for y in range(total_h):
        t   = y / total_h
        a   = int(200 - t * 80)
        draw.line([(0, y), (vid_w, y)], fill=(8, 10, 18, a))

    draw.line([(0, total_h), (vid_w, total_h)], fill=(*C_BORDER, 180), width=1)

    f_hud  = _font(int(H * 0.024))
    f_time = _font(int(H * 0.026))
    pad    = int(H * 0.012)

    # 영상 이름 (좌측, 작은 아이콘 포함)
    dot_x, dot_y = pad*2 + 4, bar_h//2
    for r, c in [(4,(70,100,200,30)),(3,(90,130,220,60)),(2,(130,170,255,180))]:
        draw.ellipse([(dot_x-r,dot_y-r),(dot_x+r,dot_y+r)], fill=c)
    draw.text((dot_x+10, (bar_h - int(H*0.024))//2),
              video_id, font=f_hud, fill=(*C_HUD_TEXT, 220))

    # 타임스탬프 (우측)
    t_str = f"{int(t_sec)//60:02d}:{t_sec%60:05.2f}"
    tw    = _text_w(draw, t_str, f_time)
    tx    = vid_w - tw - pad*2
    ty    = (bar_h - int(H*0.026)) // 2
    draw.text((tx+1, ty+1), t_str, font=f_time, fill=(0,0,0,120))
    draw.text((tx, ty),     t_str, font=f_time, fill=(200, 215, 255, 240))

    # ── 진행바 ────────────────────────────────────────────────────────────────
    py0   = bar_h
    py1   = bar_h + prog_h
    # 배경
    draw.rectangle([(0, py0), (vid_w, py1)], fill=(*C_PROG_BG, 230))

    # 진행 채우기
    if total_sec > 0:
        prog  = min(1.0, t_sec / total_sec)
        fill_w = int(vid_w * prog)
        if fill_w > 0:
            # 그라디언트 진행바
            for x in range(fill_w):
                t_   = x / max(fill_w, 1)
                r_   = int(50 + t_*40)
                g_   = int(100 + t_*20)
                b_   = int(200 + t_*20)
                draw.line([(x, py0+1), (x, py1-1)], fill=(r_,g_,b_, 220))

        # 이벤트 마커
        for et in event_times:
            ex = int(vid_w * et / total_sec)
            draw.line([(ex, py0), (ex, py1)], fill=(*C_PROG_EVT, 200), width=2)

        # 핸들 원
        hx = fill_w
        hy = (py0+py1)//2
        draw.ellipse([(hx-3, hy-3),(hx+3, hy+3)], fill=(180,210,255,255))

    result = cv2.cvtColor(np.array(img.convert("RGB")), cv2.COLOR_RGB2BGR)
    return result


# ── 메인 렌더 ────────────────────────────────────────────────────────────────────
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

    # CSV 로드 (중앙값 + 이벤트 시간 추출)
    frame_csv = FRAME_DIR / f"{video_id}.csv"
    event_times = []
    if frame_csv.exists():
        df = pd.read_csv(frame_csv)
        med_yaw   = float(df["head_yaw"].median())
        med_pitch = float(df["head_pitch"].median())
        # 이벤트 타임스탬프 (있다면)
        if "timestamp" in df.columns and "event" in df.columns:
            evt_df = df[df["event"].notna() & (df["event"] != "normal")]
            event_times = evt_df["timestamp"].tolist()
    else:
        med_yaw = med_pitch = 0.0
    print(f"  median — yaw:{med_yaw:.1f}°  pitch:{med_pitch:.1f}°")
    print(f"  이벤트 마커: {len(event_times)}개")

    cap       = cv2.VideoCapture(str(video_path))
    fps       = cap.get(cv2.CAP_PROP_FPS) or 30.0
    W         = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H         = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    total_sec    = total_frames / fps
    dist      = np.zeros((4,1), dtype=np.float64)
    cam       = _camera_matrix(W, H)

    buf_n     = int(GRAPH_SECS * fps)
    buf_yaw   = deque([0.0]*buf_n, maxlen=buf_n)
    buf_pitch = deque([0.0]*buf_n, maxlen=buf_n)
    buf_ex    = deque([0.5]*buf_n, maxlen=buf_n)

    RENDER_DIR.mkdir(parents=True, exist_ok=True)
    out_path  = RENDER_DIR / f"{video_id}_analysis.mp4"
    out       = cv2.VideoWriter(str(out_path),
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
                    over     = yaw_over or pit_over

                    # 글로우 박스
                    draw_face_box_glow(frame, x1, y1, x2, y2, over)

                    image_pts = np.array([kps[2],kps[0],kps[1],kps[3],kps[4]], np.float64)
                    _, rvec, tvec = cv2.solvePnP(
                        _MODEL_POINTS, image_pts, cam, dist, flags=cv2.SOLVEPNP_EPNP)
                    draw_head_axes(frame, rvec, tvec, cam, dist, kps[2])

                    buf_yaw.append(d_yaw)
                    buf_pitch.append(d_pitch)
                    any_over = over

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

        # 상단 HUD (진행바 포함)
        t = fidx / fps
        frame = draw_hud(frame, t, total_sec, video_id, any_over, event_times)

        out.write(frame)
        fidx += 1
        if fidx % 30 == 0:
            print(f"  {fidx}/{total_frames} frames ({t:.1f}s)...", end="\r")

    cap.release()
    out.release()
    mesh.close()
    print(f"\n  [완료] {out_path.name}  ({fidx} frames)")
    print(f"  경로: {out_path.resolve()}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    args = parser.parse_args()
    print(f"\n분석 시각화 영상 생성 (v2): {args.video}")
    render(Path(args.video))


if __name__ == "__main__":
    main()
