#!/usr/bin/env python3
"""
render_warning_video.py
=======================
분석 결과를 영상에 경고 오버레이로 합성해 내보낸다.

- 얼굴 바운딩박스: 정상=초록, 경고=빨강
- 이벤트 복귀 직후 하단 중앙 '부정행위 의심!' 배너

사용법:
  python render_warning_video.py --folder data/test_videos/
  python render_warning_video.py --video data/test_videos/2\ 좌+우\ 반복.mov
"""
import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

FFMPEG = shutil.which("ffmpeg") or "/opt/miniconda3/envs/ai_env/bin/ffmpeg"

OUTPUT_DIR  = Path("data/results/test")
RENDER_DIR  = Path("data/results/rendered")
FRAME_DIR   = OUTPUT_DIR   # frame CSV 도 같은 폴더에 저장됨

FONT_PATH   = "/Library/Fonts/AppleSDGothicNeo.ttc"
FONT_IDX    = 0
WARN_LINGER = 1.5   # 이벤트 종료 후 경고 유지 (초)

# 바운딩박스 색 (BGR)
COLOR_NORMAL  = (50, 220, 50)    # 초록
COLOR_WARNING = (50, 50, 220)    # 빨강
BOX_THICK     = 3


def _get_font(size: int):
    try:
        return ImageFont.truetype(FONT_PATH, size, index=FONT_IDX)
    except Exception:
        return ImageFont.load_default()


def _draw_warning_banner(frame_bgr: np.ndarray) -> np.ndarray:
    """하단 중앙에 '부정행위 의심!' 배너"""
    h, w = frame_bgr.shape[:2]
    img  = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img)

    font = _get_font(int(h * 0.058))
    text = "부정행위 의심!"

    bbox   = draw.textbbox((0, 0), text, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    pad    = int(h * 0.022)
    banner_h = th + pad * 2
    y0     = h - banner_h

    overlay = img.copy()
    ImageDraw.Draw(overlay).rectangle([(0, y0), (w, h)], fill=(10, 10, 10))
    img  = Image.blend(img, overlay, alpha=0.75)
    draw = ImageDraw.Draw(img)

    draw.rectangle([(0, y0), (w, y0 + int(h * 0.005))], fill=(220, 50, 50))

    tx = (w - tw) // 2
    ty = y0 + pad
    draw.text((tx + 2, ty + 2), text, font=font, fill=(80, 0, 0))
    draw.text((tx, ty),         text, font=font, fill=(255, 75, 75))

    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


def _draw_bbox(frame: np.ndarray, x1: int, y1: int, x2: int, y2: int,
               warning: bool) -> np.ndarray:
    """얼굴 바운딩박스 — 정상=초록, 경고=빨강"""
    color = COLOR_WARNING if warning else COLOR_NORMAL
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, BOX_THICK)
    return frame


def _build_bbox_lookup(frame_csv: Path, native_fps: float) -> dict:
    """frame CSV → {frame_idx: (x1,y1,x2,y2)} 딕셔너리"""
    if not frame_csv.exists():
        return {}
    df = pd.read_csv(frame_csv)
    df = df[df["face_detected"] == 1]
    lookup = {}
    for _, row in df.iterrows():
        t = float(row["time_sec"])
        fidx = round(t * native_fps)
        try:
            lookup[fidx] = (int(row["bbox_x1"]), int(row["bbox_y1"]),
                            int(row["bbox_x2"]), int(row["bbox_y2"]))
        except Exception:
            pass
    return lookup


def render_video(video_path: Path, result_csv: Path,
                 frame_csv: Path, output_path: Path):
    # 이벤트 로드
    events = []
    if result_csv.exists():
        df = pd.read_csv(result_csv)
        for _, row in df.iterrows():
            events.append({
                "start": float(row["start"]),
                "end":   float(row["end"]),
                "type":  str(row["event_type"]),
            })

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"  [ERR] 열기 실패: {video_path.name}")
        return False

    fps    = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w      = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h      = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"avc1")
    out    = cv2.VideoWriter(str(output_path), fourcc, fps, (w, h))

    # bbox 룩업 테이블 (frame_idx → bbox)
    bbox_lookup = _build_bbox_lookup(frame_csv, fps)

    # bbox 룩업은 샘플링된 프레임만 있으므로 가장 가까운 값 보간
    if bbox_lookup:
        sorted_keys = sorted(bbox_lookup.keys())

    def _nearest_bbox(fidx):
        if not bbox_lookup:
            return None
        lo, hi = 0, len(sorted_keys) - 1
        while lo < hi:
            mid = (lo + hi) // 2
            if sorted_keys[mid] < fidx:
                lo = mid + 1
            else:
                hi = mid
        best = sorted_keys[lo]
        return bbox_lookup[best]

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        t = frame_idx / fps

        # 경고 활성 여부 (시각 이벤트만 기준)
        warning = any(
            ev["end"] <= t <= ev["end"] + WARN_LINGER
            for ev in events if ev["type"] != "external_voice"
        )

        # 경고 배너
        if warning:
            frame = _draw_warning_banner(frame)

        out.write(frame)
        frame_idx += 1

    cap.release()
    out.release()

    # 원본 오디오 합치기 (ffmpeg)
    tmp = output_path.with_suffix(".tmp.mp4")
    output_path.rename(tmp)
    try:
        subprocess.run([
            FFMPEG, "-y",
            "-i", str(tmp),
            "-i", str(video_path),
            "-map", "0:v:0",
            "-map", "1:a:0?",
            "-c:v", "copy",
            "-c:a", "aac",
            "-shortest",
            str(output_path)
        ], check=True, capture_output=True)
        tmp.unlink()
    except Exception as e:
        print(f"  [경고] 오디오 합성 실패: {e}")
        tmp.rename(output_path)

    print(f"  [저장] {output_path.name}  ({frame_idx} frames, 오디오 포함)")
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", help="테스트 영상 폴더")
    parser.add_argument("--video",  help="단일 영상 경로")
    args = parser.parse_args()

    RENDER_DIR.mkdir(parents=True, exist_ok=True)

    summary_csv = OUTPUT_DIR / "_summary.csv"
    if not summary_csv.exists():
        print("[ERROR] 분석 결과 없음. 먼저 run_test_videos.py 실행 필요")
        sys.exit(1)

    if args.video:
        videos = [Path(args.video)]
    elif args.folder:
        folder = Path(args.folder)
        videos = sorted(list(folder.glob("*.mp4")) + list(folder.glob("*.mov")))
    else:
        print("[ERROR] --folder 또는 --video 필요")
        sys.exit(1)

    print(f"\n경고 영상 렌더링 — {len(videos)}개")
    print("=" * 50)

    for video_path in videos:
        video_id   = video_path.stem
        result_csv = OUTPUT_DIR / f"{video_id}_result.csv"
        frame_csv  = OUTPUT_DIR / f"{video_id}.csv"
        output_mp4 = RENDER_DIR / f"{video_id}_warning.mp4"
        print(f"\n  [{video_id}]")
        render_video(video_path, result_csv, frame_csv, output_mp4)

    print(f"\n완료 → {RENDER_DIR.resolve()}")


if __name__ == "__main__":
    main()
