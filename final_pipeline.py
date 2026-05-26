#!/usr/bin/env python3
"""
final_pipeline.py
=================
검증 영상 1개를 받아 전체 이상 탐지 파이프라인을 실행한다.

모듈 3개가 통합 동작:
  [1] 시각 분석  — head_yaw/pitch + eye tracking → event 탐지 + 반복 패턴
  [2] 객체 탐지  — YOLOv11 → 책/핸드폰/태블릿/노트북 감지
  [3] 음성 분석  — resemblyzer → 외부 목소리 감지

사용법:
  python final_pipeline.py --video path/to/video.mp4

출력:
  data/results/<video_id>_final_report.csv  ← 시간대별 이상 이벤트 전체
  data/results/<video_id>_summary.txt       ← 요약 리포트
"""
import argparse
import sys
from pathlib import Path

import pandas as pd
import cv2

# 프로젝트 루트를 경로에 추가
sys.path.insert(0, str(Path(__file__).parent))

from extract_features    import process_video
from event_detection     import load_normal_ranges, load_per_video_baseline, detect_events_for_video
from repetition_analysis import analyze_video_repetition
from audio_analysis      import analyze_external_voice

# ─────────────────────────────────────────────────────────────────
# 설정
# ─────────────────────────────────────────────────────────────────
NORMAL_RANGE_PATH   = Path("data/model/normal_ranges.csv")
BASELINE_PATH       = Path("data/model/per_video_baseline.csv")
YOLO_CONF  = 0.35
YOLO_IOU   = 0.45
OUTPUT_DIR = Path("data/results")

# COCO 기본 모델 사용 (핸드폰·책·노트북·태블릿 포함)
YOLO_COCO_MODEL = Path("data/yolo_weights/yolo11n.pt")   # 없으면 자동 다운로드
COCO_CLASSES = {
    63: "laptop",
    67: "phone",
    73: "book",
    # tablet은 COCO에 없음 → 커스텀 모델 병행
}
YOLO_MODELS_CUSTOM = {
    "tablet": Path("data/yolo_weights/tablet_best.pt"),
}
YOLO_KO = {"book": "책", "phone": "핸드폰", "tablet": "태블릿", "laptop": "노트북"}

# ─────────────────────────────────────────────────────────────────
# 모듈 1: 시각 분석 (기존 파이프라인)
# ─────────────────────────────────────────────────────────────────
def run_visual_analysis(video_path: str, video_id: str) -> dict:
    """head/eye event + 반복 패턴 분석"""
    print("\n  [1/3] 시각 분석 (head pose + eye tracking)...")

    # 피처 추출
    frame_csv_path = Path(f"data/frame_csv/{video_id}.csv")
    if not frame_csv_path.exists():
        print(f"    프레임 피처 추출 중...")
        process_video(video_path, str(frame_csv_path))
    else:
        print(f"    기존 프레임 CSV 사용: {frame_csv_path.name}")

    df_frames = pd.read_csv(frame_csv_path, dtype={"video_id": str})

    # 정상 범위 + 기준선 로드
    ranges    = load_normal_ranges(NORMAL_RANGE_PATH)
    baselines = load_per_video_baseline(BASELINE_PATH)

    # 이벤트 탐지
    events_df = detect_events_for_video(df_frames, video_id, ranges, baselines)

    # 반복 패턴 분석
    rep_result = {}
    if not events_df.empty:
        rep_result = analyze_video_repetition(events_df, video_id)

    return {
        "frame_count":     len(df_frames),
        "face_detected":   int(df_frames["face_detected"].sum()),
        "events":          events_df,
        "repetition":      rep_result,
    }


# ─────────────────────────────────────────────────────────────────
# 모듈 2: 객체 탐지 (YOLOv11)
# ─────────────────────────────────────────────────────────────────
def run_object_detection(video_path: str, fps_sample: int = 5) -> list[dict]:
    """COCO 기본 모델(핸드폰·책·노트북) + 태블릿 커스텀 모델"""
    print("\n  [2/3] 객체 탐지 (YOLOv11)...")

    from ultralytics import YOLO

    # COCO 모델 로드 (없으면 자동 다운로드)
    coco_model = YOLO(str(YOLO_COCO_MODEL) if YOLO_COCO_MODEL.exists() else "yolo11n.pt")

    # 태블릿 커스텀 모델
    custom_models = {}
    for name, p in YOLO_MODELS_CUSTOM.items():
        if p.exists():
            custom_models[name] = YOLO(str(p))

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"    [WARN] 영상 열기 실패: {video_path}")
        return []

    video_fps  = cap.get(cv2.CAP_PROP_FPS) or 30
    frame_step = max(1, int(video_fps / fps_sample))
    detections = []
    frame_idx  = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_step == 0:
            time_sec = frame_idx / video_fps

            # COCO 모델
            results = coco_model(frame, conf=YOLO_CONF, iou=YOLO_IOU, verbose=False)
            for r in results:
                for box in r.boxes:
                    cls_id = int(box.cls[0])
                    if cls_id in COCO_CLASSES:
                        detections.append({
                            "time_sec":   round(time_sec, 3),
                            "class_name": COCO_CLASSES[cls_id],
                            "confidence": round(float(box.conf[0]), 4),
                            "event_type": f"object_{COCO_CLASSES[cls_id]}",
                        })

            # 커스텀 모델 (태블릿)
            for class_name, model in custom_models.items():
                results = model(frame, conf=YOLO_CONF, iou=YOLO_IOU, verbose=False)
                for r in results:
                    for box in r.boxes:
                        detections.append({
                            "time_sec":   round(time_sec, 3),
                            "class_name": class_name,
                            "confidence": round(float(box.conf[0]), 4),
                            "event_type": f"object_{class_name}",
                        })

        frame_idx += 1

    cap.release()

    events = _merge_object_events(detections)
    print(f"    감지된 객체 이벤트: {len(events)}건")
    return events


def _merge_object_events(detections: list[dict],
                          min_duration: float = 0.5,
                          gap_tolerance: float = 0.6) -> list[dict]:
    """연속된 동일 클래스 탐지를 하나의 이벤트로 묶는다."""
    if not detections:
        return []

    from itertools import groupby
    events = []

    # 클래스별로 처리
    by_class = {}
    for d in detections:
        by_class.setdefault(d["class_name"], []).append(d)

    for cls_name, dets in by_class.items():
        dets.sort(key=lambda x: x["time_sec"])
        seg_start = dets[0]["time_sec"]
        seg_end   = dets[0]["time_sec"]
        seg_conf  = dets[0]["confidence"]

        for d in dets[1:]:
            if d["time_sec"] <= seg_end + gap_tolerance:
                seg_end  = d["time_sec"]
                seg_conf = max(seg_conf, d["confidence"])
            else:
                dur = seg_end - seg_start
                if dur >= min_duration:
                    events.append({
                        "start":      round(seg_start, 2),
                        "end":        round(seg_end,   2),
                        "duration":   round(dur,        2),
                        "class_name": cls_name,
                        "confidence": round(seg_conf,   4),
                        "event_type": f"object_{cls_name}",
                    })
                seg_start = d["time_sec"]
                seg_end   = d["time_sec"]
                seg_conf  = d["confidence"]

        dur = seg_end - seg_start
        if dur >= min_duration:
            events.append({
                "start":      round(seg_start, 2),
                "end":        round(seg_end,   2),
                "duration":   round(dur,        2),
                "class_name": cls_name,
                "confidence": round(seg_conf,   4),
                "event_type": f"object_{cls_name}",
            })

    events.sort(key=lambda x: x["start"])
    return events


# ─────────────────────────────────────────────────────────────────
# 모듈 3: 음성 분석
# ─────────────────────────────────────────────────────────────────
def run_audio_analysis(video_path: str) -> list[dict]:
    """resemblyzer 기반 외부 목소리 감지"""
    print("\n  [3/3] 음성 분석 (외부 목소리 탐지)...")
    events = analyze_external_voice(video_path)
    print(f"    외부 목소리 이벤트: {len(events)}건")
    return events


# ─────────────────────────────────────────────────────────────────
# 결과 통합 및 저장
# ─────────────────────────────────────────────────────────────────
def merge_and_save(video_id: str,
                   visual_result: dict,
                   object_events: list,
                   audio_events:  list) -> Path:
    """3개 모듈 결과를 하나의 CSV로 통합 저장"""

    all_events = []

    # 시각 이벤트
    if not visual_result["events"].empty:
        ev = visual_result["events"].copy()
        ev["module"] = "visual"
        all_events.append(ev[["start_time","end_time","duration_sec","event_type","module"]
                              ].rename(columns={"start_time":"start","end_time":"end","duration_sec":"duration"}))

    # 객체 이벤트
    if object_events:
        ev = pd.DataFrame(object_events)
        ev["module"] = "object"
        all_events.append(ev[["start","end","duration","event_type","module"]])

    # 음성 이벤트
    if audio_events:
        ev = pd.DataFrame(audio_events)
        ev["module"] = "audio"
        all_events.append(ev[["start","end","duration","event_type","module"]])

    if all_events:
        combined = pd.concat(all_events, ignore_index=True)
        combined = combined.sort_values("start").reset_index(drop=True)
        combined.insert(0, "video_id", video_id)
    else:
        combined = pd.DataFrame(columns=["video_id","start","end","duration","event_type","module"])

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_csv = OUTPUT_DIR / f"{video_id}_final_report.csv"
    combined.to_csv(out_csv, index=False)
    return out_csv, combined


def print_summary(video_id: str, visual: dict, obj: list, aud: list, combined: pd.DataFrame):
    """콘솔 요약 리포트"""
    rep = visual.get("repetition", {})

    print("\n" + "=" * 65)
    print(f"  최종 분석 결과 — {video_id}")
    print("=" * 65)

    print(f"\n  [시각] 이벤트 {len(visual['events'])}건")
    if not visual["events"].empty:
        for _, row in visual["events"].iterrows():
            print(f"    {row['event_type']:<20} {row['start_time']:.1f}s ~ {row['end_time']:.1f}s  ({row['duration_sec']:.1f}초)")

    if rep:
        print(f"\n  [반복 패턴]")
        print(f"    우측 {rep.get('right',0)}회 / 좌측 {rep.get('left',0)}회 / 하향 {rep.get('down',0)}회")
        print(f"    버스트 {rep.get('bursts',0)}회 / 교차패턴 {rep.get('switches',0)}회")
        score = rep.get('suspicion_score', 0)
        print(f"    의심도 점수: {score}")

    print(f"\n  [객체] 이벤트 {len(obj)}건")
    for ev in obj:
        ko = YOLO_KO.get(ev.get("class_name",""), ev.get("class_name",""))
        print(f"    {ko:<8} {ev['start']:.1f}s ~ {ev['end']:.1f}s  (신뢰도 {ev['confidence']:.2f})")

    print(f"\n  [음성] 외부 목소리 {len(aud)}건")
    for ev in aud:
        print(f"    {ev['start']:.1f}s ~ {ev['end']:.1f}s  (유사도 {ev['similarity']:.3f})")

    total = len(combined)
    print(f"\n  총 이상 이벤트: {total}건")
    if total == 0:
        print("  → 이상 없음")
    elif total <= 2:
        print("  → 경미한 이상 감지, 면접관 확인 권장")
    else:
        print("  → 부정행위 의심! 면접관 즉시 확인 필요")
    print("=" * 65)


# ─────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="비대면 면접 이상 탐지 최종 파이프라인")
    parser.add_argument("--video", required=True, help="분석할 영상 경로")
    args = parser.parse_args()

    video_path = args.video
    if not Path(video_path).exists():
        print(f"[ERROR] 영상 없음: {video_path}")
        sys.exit(1)

    video_id = Path(video_path).stem
    print(f"\n분석 시작: {video_id}")
    print("=" * 65)

    # 3개 모듈 순차 실행
    visual_result  = run_visual_analysis(video_path, video_id)
    object_events  = run_object_detection(video_path)
    audio_events   = run_audio_analysis(video_path)

    # 결과 통합
    out_csv, combined = merge_and_save(video_id, visual_result, object_events, audio_events)

    # 요약 출력
    print_summary(video_id, visual_result, object_events, audio_events, combined)
    print(f"\n  리포트 저장 → {out_csv.resolve()}")


if __name__ == "__main__":
    main()
