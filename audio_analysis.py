#!/usr/bin/env python3
"""
audio_analysis.py
=================
영상에서 오디오를 추출하고 resemblyzer 기반 화자 분리로
면접자 외 외부 목소리를 감지한다.

원리:
  1. 영상 앞 N초 = 면접자 목소리 기준 임베딩으로 저장
  2. 이후 구간을 1초 단위로 분할
  3. 각 구간의 임베딩과 기준 임베딩의 코사인 유사도 계산
  4. 유사도 < SIMILARITY_THRESHOLD → 외부 목소리 FLAG

출력:
  audio_events: [{"start": float, "end": float, "similarity": float}]
  → 외부 목소리가 감지된 시간 구간 목록
"""
import os
import subprocess
import tempfile
from pathlib import Path

import numpy as np

# ─────────────────────────────────────────────────────────────────
# 설정
# ─────────────────────────────────────────────────────────────────
REFERENCE_DURATION   = 10.0   # 기준 목소리로 사용할 영상 앞부분 (초)
SEGMENT_DURATION     = 1.0    # 분석 단위 (초)
SIMILARITY_THRESHOLD = 0.70   # 이 값보다 낮으면 외부 목소리로 판정
MIN_FLAG_DURATION    = 2.0    # 최소 연속 감지 시간 (초) — 짧은 노이즈 제거
SAMPLE_RATE          = 16000  # resemblyzer 요구 샘플레이트


# ─────────────────────────────────────────────────────────────────
# 오디오 추출
# ─────────────────────────────────────────────────────────────────
def extract_audio(video_path: str, output_wav: str, sr: int = SAMPLE_RATE) -> bool:
    """
    ffmpeg 으로 영상에서 오디오를 추출해 wav로 저장한다.
    오디오 트랙이 없으면 False 반환.
    """
    import shutil
    ffmpeg_bin = shutil.which("ffmpeg") or "/opt/miniconda3/envs/ai_env/bin/ffmpeg"
    cmd = [
        ffmpeg_bin, "-y",
        "-i", video_path,
        "-vn",
        "-acodec", "pcm_s16le",
        "-ar", str(sr),
        "-ac", "1",
        output_wav,
        "-loglevel", "error",
    ]
    result = subprocess.run(cmd, capture_output=True)
    if result.returncode != 0:
        return False
    return Path(output_wav).exists() and Path(output_wav).stat().st_size > 0


# ─────────────────────────────────────────────────────────────────
# 화자 분리 분석
# ─────────────────────────────────────────────────────────────────
def analyze_external_voice(video_path: str) -> list[dict]:
    """
    영상에서 외부 목소리 구간을 감지한다.

    반환:
        [{"start": float, "end": float, "similarity": float}, ...]
        → 외부 목소리로 판정된 시간 구간 목록
        → 빈 리스트: 외부 목소리 없음 또는 오디오 없음
    """
    try:
        from resemblyzer import VoiceEncoder, preprocess_wav
    except ImportError:
        raise ImportError("pip install resemblyzer 를 먼저 실행하세요.")

    # ── 1. 오디오 추출 ───────────────────────────────────────────
    with tempfile.TemporaryDirectory() as tmpdir:
        wav_path = os.path.join(tmpdir, "audio.wav")

        if not extract_audio(video_path, wav_path):
            print(f"  [WARN] {Path(video_path).name}: 오디오 트랙 없음 → 음성 분석 생략")
            return []

        try:
            wav = preprocess_wav(wav_path)
        except Exception as e:
            print(f"  [WARN] 오디오 전처리 실패: {e}")
            return []

        total_duration = len(wav) / SAMPLE_RATE
        if total_duration < REFERENCE_DURATION + SEGMENT_DURATION:
            print(f"  [WARN] 영상이 너무 짧아 음성 분석 생략 ({total_duration:.1f}초)")
            return []

        encoder = VoiceEncoder()

        # ── 2. 기준 임베딩 추출 (영상 앞 N초) ───────────────────
        ref_samples = int(REFERENCE_DURATION * SAMPLE_RATE)
        ref_wav     = wav[:ref_samples]

        # 목소리 활동 확인 (무음 구간이면 기준으로 쓸 수 없음)
        if np.abs(ref_wav).mean() < 0.001:
            print("  [WARN] 기준 구간이 무음입니다. 음성 분석 생략.")
            return []

        ref_embedding = encoder.embed_utterance(ref_wav)

        # ── 3. 구간별 유사도 계산 ────────────────────────────────
        seg_samples = int(SEGMENT_DURATION * SAMPLE_RATE)
        start_sample = int(REFERENCE_DURATION * SAMPLE_RATE)

        similarities = []   # (start_sec, end_sec, similarity)
        t = REFERENCE_DURATION

        while start_sample + seg_samples <= len(wav):
            seg = wav[start_sample: start_sample + seg_samples]

            # 무음 구간 스킵
            if np.abs(seg).mean() < 0.001:
                similarities.append((t, t + SEGMENT_DURATION, 1.0))  # 무음 = 정상
                start_sample += seg_samples
                t += SEGMENT_DURATION
                continue

            emb = encoder.embed_utterance(seg)
            sim = float(np.dot(ref_embedding, emb) /
                        (np.linalg.norm(ref_embedding) * np.linalg.norm(emb) + 1e-9))
            similarities.append((t, t + SEGMENT_DURATION, sim))
            start_sample += seg_samples
            t += SEGMENT_DURATION

        # ── 4. 외부 목소리 구간 추출 (연속 구간 묶기) ───────────
        flag_segments = [(s, e, sim) for s, e, sim in similarities
                         if sim < SIMILARITY_THRESHOLD]

        if not flag_segments:
            return []

        # 연속된 1초 구간 묶기
        events = []
        seg_start, seg_end, seg_sim = flag_segments[0]

        for (s, e, sim) in flag_segments[1:]:
            if s <= seg_end + 0.1:   # 연속 (0.1초 여유)
                seg_end = e
                seg_sim = min(seg_sim, sim)
            else:
                dur = seg_end - seg_start
                if dur >= MIN_FLAG_DURATION:
                    events.append({
                        "start":      round(seg_start, 2),
                        "end":        round(seg_end,   2),
                        "duration":   round(dur,        2),
                        "similarity": round(seg_sim,    4),
                        "event_type": "external_voice",
                    })
                seg_start, seg_end, seg_sim = s, e, sim

        # 마지막 구간
        dur = seg_end - seg_start
        if dur >= MIN_FLAG_DURATION:
            events.append({
                "start":      round(seg_start, 2),
                "end":        round(seg_end,   2),
                "duration":   round(dur,        2),
                "similarity": round(seg_sim,    4),
                "event_type": "external_voice",
            })

        return events


# ─────────────────────────────────────────────────────────────────
# 배치 처리 (학습 데이터 전체 분석)
# ─────────────────────────────────────────────────────────────────
def analyze_all_videos(video_dir: Path, output_path: Path):
    """
    video_dir 내 모든 mp4를 분석해 결과를 CSV로 저장한다.
    정상 데이터에서 외부 목소리 사건 수를 기록한다 (베이스라인).
    """
    import pandas as pd

    files = sorted(video_dir.glob("*.mp4"))
    print(f"\n{len(files)}개 영상 음성 분석 중...")

    all_events = []
    for i, f in enumerate(files, 1):
        video_id = f.stem
        events = analyze_external_voice(str(f))
        for ev in events:
            ev["video_id"] = video_id
            all_events.append(ev)
        status = f"{len(events)}개 외부음성 감지" if events else "정상"
        print(f"  [{i:3d}/{len(files)}] {video_id}  →  {status}")

    df = pd.DataFrame(all_events) if all_events else pd.DataFrame(
        columns=["video_id", "start", "end", "duration", "similarity", "event_type"]
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    print(f"\n저장 완료 → {output_path.resolve()}")
    print(f"총 외부 음성 이벤트: {len(df)}건  ({df['video_id'].nunique() if len(df) else 0}개 영상)")
    return df


# ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    VIDEO_DIR   = Path("data/raw_videos")
    OUTPUT_PATH = Path("data/results/audio_events.csv")
    analyze_all_videos(VIDEO_DIR, OUTPUT_PATH)
