#!/usr/bin/env python3
"""
event_detection.py
==================
프레임 단위 CSV와 정상 범위(normal_ranges.csv), 영상별 중앙값 기준선
(per_video_baseline.csv)을 바탕으로 연속된 응시 편향 구간(event)을 추출한다.

핵심 설계 원칙:
  - '총 몇 초 봤는지'보다 '한 번에 몇 초 지속됐는지'가 더 중요하다.
  - 연속된 flag 프레임을 하나의 event로 묶고,
    start_time / end_time / duration_sec 를 기록한다.
  - duration_sec < MIN_DURATION 인 짧은 노이즈 event는 제거한다.

──────────────────────────────────────────────────
[카메라 각도 보정 — per-video median baseline]

  영상마다 카메라 위치·각도가 달라 head_yaw 절대값에 구조적 오프셋
  (수 도 ~ 십수 도)이 존재한다. 절대값 기준 flag를 쓰면 카메라가
  약간 옆에 있는 영상에서 false positive가 증가한다.

  해결: 각 영상의 median을 개인 기준선으로 사용.
    delta_yaw   = head_yaw   - video_median_yaw
    delta_pitch = head_pitch - video_median_pitch
    delta_ex    = eye_x_ratio - video_median_eye_x_ratio
    delta_ey    = eye_y_ratio - video_median_eye_y_ratio

  Flag 조건:
    right_flag : delta_yaw  > +thr_delta_yaw  OR  delta_ex > +thr_delta_ex
    left_flag  : delta_yaw  < -thr_delta_yaw  OR  delta_ex < -thr_delta_ex
    down_flag  : delta_pitch > +thr_delta_pitch OR  delta_ey > +thr_delta_ey

  thr_delta = 2.0 × 전체집단 IQR  (compute_normal_ranges.py 에서 계산)
    → IQR은 이상치에 강건한 산포 지표 (breakdown point 25%)
    → k=2.0: 정상 분포의 중심 95.4% 포용 (IQR ≈ 1.35σ → 2×IQR ≈ 2.7σ)

  영상 내 baseline이 없으면(CSV 없음 등) 절대값 기준 thr_lo/thr_hi 사용.
──────────────────────────────────────────────────

입력:
  data/frame_csv/*.csv
  data/model/normal_ranges.csv
  data/model/per_video_baseline.csv

출력:
  data/event_csv/<video_id>_event.csv  (영상별)
  data/event_csv/_all_events.csv       (전체 합본)

출력 컬럼:
  video_id, event_id, event_type,
  start_frame, end_frame, start_time, end_time, duration_sec, frame_count

event_type:
  right_reference  – 우측 편향 (delta_yaw > +thr  OR  delta_ex > +thr)
  left_reference   – 좌측 편향 (delta_yaw < -thr  OR  delta_ex < -thr)
  down_reference   – 하향 편향 (delta_pitch > +thr OR  delta_ey > +thr)
"""

import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
import pandas as pd
import numpy as np

# ══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════════════

FRAME_CSV_DIR  = Path("data/frame_csv")
NORMAL_RANGE   = Path("data/model/normal_ranges.csv")
BASELINE_PATH  = Path("data/model/per_video_baseline.csv")
OUTPUT_DIR     = Path("data/event_csv")

# 최소 지속 시간: 이보다 짧은 event는 노이즈로 제거
# 0.3s → 0.5s: 생각하다 눈 굴리거나 순간 고개 돌리는 정도(0.3~0.5s)는 자연스러운 행동
MIN_DURATION   = 0.5   # seconds

# 사용 feature
FEATURES_USED  = ["head_yaw", "head_pitch", "eye_x_ratio", "eye_y_ratio"]

EVENT_TYPES    = ["right_reference", "left_reference", "down_reference"]

# per-video baseline에 사용할 IQR 배수 (compute_normal_ranges.py와 동일값)
DELTA_K        = 2.0


# ══════════════════════════════════════════════════════════════════════════════
# 1. 정상 범위 및 per-video baseline 로드
# ══════════════════════════════════════════════════════════════════════════════

def load_normal_ranges(path: Path) -> dict:
    """
    normal_ranges.csv를 읽어 {feature: {stat: value}} 딕셔너리로 반환.
    예) ranges["head_yaw"]["thr_delta"] → float
    """
    if not path.exists():
        raise FileNotFoundError(
            f"정상 범위 파일 없음: {path}\n"
            "  → python compute_normal_ranges.py 를 먼저 실행하세요."
        )
    df = pd.read_csv(path)
    df = df.set_index("feature")
    return df.to_dict(orient="index")


def load_per_video_baseline(path: Path) -> dict:
    """
    per_video_baseline.csv를 읽어 {video_id: {col: value}} 딕셔너리로 반환.
    예) baseline["S-e9-bW4seo.004"]["median_yaw"] → float

    파일이 없으면 빈 딕셔너리를 반환한다 (절대값 fallback).
    """
    if not path.exists():
        print(f"  [WARN] per-video baseline 파일 없음: {path}")
        print("  → 절대값 기준(thr_lo/thr_hi) 으로 fallback 합니다.")
        return {}
    df = pd.read_csv(path, dtype={"video_id": str})
    df = df.set_index("video_id")
    return df.to_dict(orient="index")


# ══════════════════════════════════════════════════════════════════════════════
# 2. 프레임별 flag 계산
# ══════════════════════════════════════════════════════════════════════════════

def assign_flags(df: pd.DataFrame,
                 ranges: dict,
                 video_baseline: dict | None = None) -> pd.DataFrame:
    """
    각 프레임에 right_flag / left_flag / down_flag 를 부여한다.

    [per-video baseline 보정 모드] (video_baseline 제공 시)
    ─────────────────────────────
    카메라 각도 차이로 인한 구조적 오프셋을 제거한다.

      delta_yaw   = head_yaw   - video_median_yaw
      delta_pitch = head_pitch - video_median_pitch
      delta_ex    = eye_x_ratio - video_median_eye_x_ratio
      delta_ey    = eye_y_ratio - video_median_eye_y_ratio

    Flag 조건 (t = thr_delta = 2.0 × IQR):
      right_flag : delta_yaw  > +t_yaw  OR  delta_ex > +t_ex
      left_flag  : delta_yaw  < -t_yaw  OR  delta_ex < -t_ex
      down_flag  : delta_pitch > +t_pitch OR  delta_ey > +t_ey

    [절대값 fallback 모드] (video_baseline 없음 시)
    ─────────────────────
    right_flag : head_yaw > thr_hi OR eye_x_ratio > thr_hi
    left_flag  : head_yaw < thr_lo OR eye_x_ratio < thr_lo
    down_flag  : head_pitch > thr_hi OR eye_y_ratio > thr_hi
    """
    df = df.copy()

    # feature 컬럼을 float으로 변환
    for col in FEATURES_USED:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # face_detected == 1 인 프레임만 flag 부여 (미검출 프레임은 모두 False)
    face_mask = df["face_detected"] == 1

    def r(feat, stat):
        return ranges[feat][stat]

    if video_baseline:
        # ── per-video delta 모드 ──────────────────────────────────────────
        med_yaw   = video_baseline["median_yaw"]
        med_pitch = video_baseline["median_pitch"]
        med_ex    = video_baseline["median_eye_x_ratio"]
        med_ey    = video_baseline["median_eye_y_ratio"]

        # thr_delta: 2.0 × IQR (compute_normal_ranges.py에서 저장)
        t_yaw   = r("head_yaw",    "thr_delta")
        t_pitch = r("head_pitch",  "thr_delta")
        t_ex    = r("eye_x_ratio", "thr_delta")
        t_ey    = r("eye_y_ratio", "thr_delta")

        d_yaw   = df["head_yaw"]    - med_yaw
        d_pitch = df["head_pitch"]  - med_pitch
        d_ex    = df["eye_x_ratio"] - med_ex
        d_ey    = df["eye_y_ratio"] - med_ey

        df["right_flag"] = face_mask & (
            (d_yaw  >  t_yaw) | (d_ex >  t_ex)
        )
        df["left_flag"]  = face_mask & (
            (d_yaw  < -t_yaw) | (d_ex < -t_ex)
        )
        # head_pitch: delta > 0 = 본인 평소보다 더 숙임 = down
        # eye_y_ratio: delta > 0 = 본인 평소보다 홍채 더 아래 = down
        df["down_flag"]  = face_mask & (
            (d_pitch >  t_pitch) | (d_ey >  t_ey)
        )

    else:
        # ── 절대값 fallback 모드 ─────────────────────────────────────────
        # right_flag — thr_hi 사용 (head 각도: mean+2.5σ, eye: p97.5)
        df["right_flag"] = face_mask & (
            (df["head_yaw"]    > r("head_yaw",    "thr_hi")) |
            (df["eye_x_ratio"] > r("eye_x_ratio", "thr_hi"))
        )
        # left_flag — thr_lo 사용 (head 각도: mean-2.5σ, eye: p2.5)
        df["left_flag"]  = face_mask & (
            (df["head_yaw"]    < r("head_yaw",    "thr_lo")) |
            (df["eye_x_ratio"] < r("eye_x_ratio", "thr_lo"))
        )
        # down_flag: 양수 pitch = 고개 숙임 방향
        df["down_flag"]  = face_mask & (
            (df["head_pitch"]  > r("head_pitch",  "thr_hi")) |
            (df["eye_y_ratio"] > r("eye_y_ratio", "thr_hi"))
        )

    return df


# ══════════════════════════════════════════════════════════════════════════════
# 3. 연속 프레임 → event 묶기
# ══════════════════════════════════════════════════════════════════════════════

def extract_events(df: pd.DataFrame,
                   flag_col: str,
                   event_type: str,
                   video_id: str,
                   min_duration: float = MIN_DURATION) -> list[dict]:
    """
    flag_col이 True인 연속된 프레임들을 하나의 event로 묶는다.

    Parameters
    ----------
    df           : 프레임 DataFrame (time_sec, frame_idx 포함)
    flag_col     : 'right_flag' | 'left_flag' | 'down_flag'
    event_type   : event 유형 레이블
    video_id     : 영상 ID
    min_duration : 최소 지속 시간 (초). 미만이면 제거.

    Returns
    -------
    list of dict, 각 dict가 하나의 event row
    """
    flag_series = df[flag_col].fillna(False).astype(bool)

    if not flag_series.any():
        return []

    # 연속 그룹 번호 부여: flag가 바뀌는 시점마다 group_id 증가
    group_id = (flag_series != flag_series.shift()).cumsum()

    events = []
    for gid, grp in df[flag_series].groupby(group_id[flag_series]):
        if grp.empty:
            continue

        start_time  = float(grp["time_sec"].iloc[0])
        end_time    = float(grp["time_sec"].iloc[-1])
        duration    = end_time - start_time

        # 최소 지속 시간 필터
        if duration < min_duration:
            continue

        events.append({
            "video_id":     video_id,
            "event_type":   event_type,
            "start_frame":  int(grp["frame_idx"].iloc[0]),
            "end_frame":    int(grp["frame_idx"].iloc[-1]),
            "start_time":   round(start_time, 4),
            "end_time":     round(end_time,   4),
            "duration_sec": round(duration,   4),
            "frame_count":  len(grp),
        })

    return events


def detect_events_for_video(df: pd.DataFrame,
                             video_id: str,
                             ranges: dict,
                             per_video_baselines: dict | None = None) -> pd.DataFrame:
    """
    단일 영상의 프레임 DataFrame에서 전체 event 목록을 추출한다.
    event_id는 영상 내에서 start_time 순으로 1부터 부여한다.

    per_video_baselines: {video_id: {...}} 형태, None이면 절대값 fallback
    """
    video_baseline = (per_video_baselines or {}).get(video_id)
    df_flagged = assign_flags(df, ranges, video_baseline=video_baseline)

    all_events = []
    flag_map = {
        "right_reference": "right_flag",
        "left_reference":  "left_flag",
        "down_reference":  "down_flag",
    }

    for event_type, flag_col in flag_map.items():
        evs = extract_events(df_flagged, flag_col, event_type, video_id)
        all_events.extend(evs)

    if not all_events:
        return pd.DataFrame(columns=[
            "video_id", "event_id", "event_type",
            "start_frame", "end_frame",
            "start_time", "end_time", "duration_sec", "frame_count",
        ])

    result = pd.DataFrame(all_events)
    result = result.sort_values("start_time").reset_index(drop=True)
    result.insert(1, "event_id", result.index + 1)   # 1-based event_id

    return result


# ══════════════════════════════════════════════════════════════════════════════
# 4. 배치 처리
# ══════════════════════════════════════════════════════════════════════════════

def process_all_videos(csv_dir: Path,
                       ranges: dict,
                       output_dir: Path,
                       per_video_baselines: dict | None = None) -> pd.DataFrame:
    """
    data/frame_csv/ 내 모든 CSV를 처리하고
    영상별 event CSV를 저장한다.
    전체 event를 합쳐 반환한다.

    per_video_baselines: load_per_video_baseline()의 반환값 (없으면 절대값 fallback)
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(csv_dir.glob("*.csv"))

    baseline_mode = "per-video median delta" if per_video_baselines else "절대값 fallback"
    print(f"  {len(files)}개 영상 처리 중... [모드: {baseline_mode}]")

    all_results = []

    for i, f in enumerate(files, 1):
        video_id = f.stem
        try:
            df = pd.read_csv(f, dtype={"video_id": str})
        except Exception as e:
            print(f"  [WARN] {f.name}: {e}")
            continue

        event_df = detect_events_for_video(df, video_id, ranges, per_video_baselines)

        # 영상별 저장
        out_path = output_dir / f"{video_id}_event.csv"
        event_df.to_csv(out_path, index=False)

        n = len(event_df)
        # baseline 보정 여부 표시
        has_bl = (per_video_baselines or {}).get(video_id) is not None
        bl_mark = "✓" if has_bl else "!"
        status  = f"{n}개 event" if n > 0 else "event 없음"
        print(f"  [{i:3d}/{len(files)}] [{bl_mark}] {video_id}  →  {status}")

        if not event_df.empty:
            all_results.append(event_df)

    if all_results:
        combined = pd.concat(all_results, ignore_index=True)
    else:
        combined = pd.DataFrame()

    return combined


# ══════════════════════════════════════════════════════════════════════════════
# 5. 요약 리포트
# ══════════════════════════════════════════════════════════════════════════════

def print_summary(combined: pd.DataFrame) -> None:
    """전체 event 통계를 콘솔에 출력한다."""
    if combined.empty:
        print("  감지된 event가 없습니다.")
        return

    print()
    print("=" * 62)
    print("  EVENT DETECTION 요약")
    print("=" * 62)
    print(f"  총 event 수       : {len(combined):,}")
    print(f"  대상 영상 수       : {combined['video_id'].nunique()}")
    print()

    # event_type별 통계
    grp = combined.groupby("event_type")["duration_sec"]
    type_stats = pd.DataFrame({
        "count":       grp.count(),
        "mean_dur(s)": grp.mean().round(3),
        "max_dur(s)":  grp.max().round(3),
        "total_dur(s)":grp.sum().round(3),
    })
    print("  [event_type별 통계]")
    print(type_stats.to_string())
    print()

    # 가장 긴 단일 event Top 10
    top10 = combined.nlargest(10, "duration_sec")[
        ["video_id", "event_type", "start_time", "end_time", "duration_sec"]
    ]
    print("  [지속시간 상위 10개 event]")
    print(top10.to_string(index=False))
    print("=" * 62)

    # 영상별 요약
    vid_summary = combined.groupby("video_id").agg(
        total_events  =("event_id",     "max"),
        total_dur_sec =("duration_sec", "sum"),
        max_dur_sec   =("duration_sec", "max"),
    ).round(3).sort_values("total_dur_sec", ascending=False)

    summary_path = OUTPUT_DIR / "_video_event_summary.csv"
    vid_summary.reset_index().to_csv(summary_path, index=False)
    print(f"\n  영상별 요약 → {summary_path.resolve()}")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("\n[1/3] 정상 범위 로드")
    ranges = load_normal_ranges(NORMAL_RANGE)
    print(f"  로드 완료: {list(ranges.keys())}")

    print("\n[2/3] 영상별 중앙값 기준선 로드")
    per_video_baselines = load_per_video_baseline(BASELINE_PATH)
    if per_video_baselines:
        print(f"  {len(per_video_baselines)}개 영상 기준선 로드 완료")
        print("  → per-video median delta 모드로 동작합니다.")
    else:
        print("  → 절대값 기준(thr_lo/thr_hi) fallback 모드로 동작합니다.")

    print("\n[3/3] event 추출 및 저장")
    combined = process_all_videos(
        FRAME_CSV_DIR, ranges, OUTPUT_DIR,
        per_video_baselines=per_video_baselines,
    )

    # 전체 합본 저장
    all_path = OUTPUT_DIR / "_all_events.csv"
    combined.to_csv(all_path, index=False)
    print(f"\n  전체 합본 → {all_path.resolve()}")

    print("\n요약 리포트")
    print_summary(combined)


if __name__ == "__main__":
    main()
