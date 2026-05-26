#!/usr/bin/env python3
"""
run_test_videos.py
==================
테스트 영상 폴더를 받아 전체 영상을 분석하고
영상별 결과 + 전체 요약을 출력한다.

객체 탐지(YOLO) 제외 — 시각 분석 + 음성 분석만 실행

사용법:
  python run_test_videos.py --folder data/test_videos/
"""
import argparse
import sys
import warnings
import os
warnings.filterwarnings("ignore")
os.environ.setdefault("GLOG_minloglevel", "3")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
from pathlib import Path

import pandas as pd
import mediapipe as mp
from insightface.app import FaceAnalysis

sys.path.insert(0, str(Path(__file__).parent))

from extract_features    import process_video, CSV_COLUMNS
from event_detection     import load_normal_ranges, load_per_video_baseline, detect_events_for_video
from repetition_analysis import analyze_video_repetition
from audio_analysis      import analyze_external_voice
from final_pipeline      import run_object_detection

NORMAL_RANGE_PATH = Path("data/model/normal_ranges.csv")
BASELINE_PATH     = Path("data/model/per_video_baseline.csv")
OUTPUT_DIR        = Path("data/results/test")

# ── 전역 모델 (한 번만 로딩) ─────────────────────────────────────
_face_app = None
_mesh     = None

def _init_models():
    global _face_app, _mesh
    if _face_app is None:
        print("  모델 로딩: InsightFace buffalo_l ...")
        _face_app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
        _face_app.prepare(ctx_id=0, det_size=(640, 640))
    if _mesh is None:
        print("  모델 로딩: MediaPipe FaceMesh ...")
        _mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=True, max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5, min_tracking_confidence=0.5,
        )
    return _face_app, _mesh


# ─────────────────────────────────────────────────────────────────
def analyze_video(video_path: Path, ranges: dict, baselines: dict) -> dict:
    video_id  = video_path.stem
    print(f"\n{'─'*60}")
    print(f"  [{video_id}]")
    print(f"{'─'*60}")

    # ── 시각 분석 ────────────────────────────────────────────────
    print("  [1/2] 시각 분석...")
    frame_csv = OUTPUT_DIR / f"{video_id}.csv"

    if not frame_csv.exists():
        face_app, mesh = _init_models()
        process_video(video_path, face_app, mesh, OUTPUT_DIR)
    else:
        print(f"    기존 프레임 CSV 사용")

    df_frames  = pd.read_csv(frame_csv, dtype={"video_id": str})
    df_frames["video_id"] = video_id
    events_df  = detect_events_for_video(df_frames, video_id, ranges, baselines)
    rep_result = analyze_video_repetition(events_df, video_id) if not events_df.empty else {}

    # ── 객체 탐지 (YOLO) ─────────────────────────────────────────
    print("  [2/3] 객체 탐지 (YOLO)...")
    yolo_events = run_object_detection(str(video_path))

    # ── 음성 분석 ────────────────────────────────────────────────
    print("  [3/3] 음성 분석...")
    audio_events = analyze_external_voice(str(video_path))

    # ── 결과 출력 ────────────────────────────────────────────────
    print(f"\n  [결과]")

    # 시각 이벤트
    if events_df.empty:
        print("    시각: 이벤트 없음")
    else:
        for _, row in events_df.iterrows():
            print(f"    시각: {row['event_type']:<20} {row['start_time']:.1f}s ~ {row['end_time']:.1f}s  ({row['duration_sec']:.1f}초)")

    # 반복 패턴
    if rep_result:
        score = rep_result.get('suspicion_score', 0)
        bursts   = rep_result.get('bursts', 0)
        switches = rep_result.get('switches', 0)
        rapids   = rep_result.get('rapids', 0)
        if score > 0:
            print(f"    반복: 버스트 {bursts}회 / 교차패턴 {switches}회 / 급속재응시 {rapids}회  → 의심도 {score}점")
        else:
            print(f"    반복: 패턴 없음")

    # 객체 탐지 이벤트
    if yolo_events:
        for ev in yolo_events:
            ko = {"book":"책","phone":"핸드폰","tablet":"태블릿","laptop":"노트북"}
            name = ko.get(ev.get("class_name",""), ev.get("class_name",""))
            print(f"    객체: {name:<8} {ev['start']:.1f}s ~ {ev['end']:.1f}s")
    else:
        print("    객체: 탐지 없음")

    # 음성 이벤트
    if audio_events:
        for ev in audio_events:
            print(f"    음성: 외부목소리  {ev['start']:.1f}s ~ {ev['end']:.1f}s  (유사도 {ev['similarity']:.3f})")
    else:
        print("    음성: 이상 없음")

    # 종합 판정
    visual_count  = len(events_df)
    max_duration  = events_df["duration_sec"].max() if not events_df.empty else 0
    score         = rep_result.get('suspicion_score', 0)
    audio_count   = len(audio_events)
    yolo_count    = len(yolo_events)

    # 부정행위 판정 조건 (하나라도 해당시 의심)
    flag_repeated  = visual_count >= 3               # 시각 이벤트 3회 이상
    flag_sustained = max_duration >= 3.0             # 단일 이벤트 3초 이상 지속
    flag_pattern   = score > 0                       # 반복 패턴 감지
    flag_audio     = audio_count >= 1                # 외부 음성 감지
    flag_object    = yolo_count >= 1                 # 객체 탐지

    print()
    if flag_repeated or flag_sustained or flag_pattern or flag_audio or flag_object:
        verdict = "부정행위 의심 — 면접관 확인 필요"
        mark    = "!!"
    else:
        verdict = "정상"
        mark    = "O"
    print(f"  [{mark}] 종합 판정: {verdict}")

    # CSV 저장
    all_rows = []
    if not events_df.empty:
        for _, r in events_df.iterrows():
            all_rows.append({"video_id": video_id, "module": "visual",
                             "start": r["start_time"], "end": r["end_time"],
                             "duration": r["duration_sec"], "event_type": r["event_type"]})
    for ev in audio_events:
        all_rows.append({"video_id": video_id, "module": "audio",
                         "start": ev["start"], "end": ev["end"],
                         "duration": ev["duration"], "event_type": "external_voice"})
    for ev in yolo_events:
        all_rows.append({"video_id": video_id, "module": "yolo",
                         "start": ev["start"], "end": ev["end"],
                         "duration": ev["duration"],
                         "event_type": f"object_{ev.get('class_name','')}"})

    if all_rows:
        pd.DataFrame(all_rows).to_csv(OUTPUT_DIR / f"{video_id}_result.csv", index=False)

    return {"video_id": video_id, "visual_events": len(events_df),
            "yolo_events": yolo_count, "audio_events": len(audio_events),
            "suspicion_score": score, "verdict": verdict}


# ─────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", required=True, help="테스트 영상 폴더 경로")
    args = parser.parse_args()

    folder = Path(args.folder)
    if not folder.exists():
        print(f"[ERROR] 폴더 없음: {folder}")
        sys.exit(1)

    videos = sorted(list(folder.glob("*.mp4")) + list(folder.glob("*.mov")))
    if not videos:
        print(f"[ERROR] mp4/mov 파일 없음: {folder}")
        sys.exit(1)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  테스트 영상 분석 시작 — {len(videos)}개 영상")
    print(f"  모드: 시각 분석 + 음성 분석 (객체탐지 제외)")
    print(f"{'='*60}")

    # 정상 범위 로드
    ranges    = load_normal_ranges(NORMAL_RANGE_PATH)
    baselines = load_per_video_baseline(BASELINE_PATH)

    # 전체 영상 분석
    summary = []
    for video_path in videos:
        result = analyze_video(video_path, ranges, baselines)
        summary.append(result)

    # 전체 요약
    print(f"\n{'='*60}")
    print("  전체 요약")
    print(f"{'='*60}")
    print(f"  {'영상':<30} {'시각':>6} {'음성':>6} {'의심도':>6}  판정")
    print(f"  {'─'*56}")
    for r in summary:
        mark = "!!" if (r['visual_events'] + r['audio_events'] + r['yolo_events'] > 0 or r['suspicion_score'] > 0) \
               else "O "
        print(f"  {mark} {r['video_id']:<28} {r['visual_events']:>6} {r['audio_events']:>6} {r['suspicion_score']:>6}")
    print(f"{'='*60}")

    pd.DataFrame(summary).to_csv(OUTPUT_DIR / "_summary.csv", index=False)
    print(f"\n  결과 저장 → {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
