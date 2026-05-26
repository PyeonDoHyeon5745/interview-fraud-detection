#!/usr/bin/env python3
"""
repetition_analysis.py
=======================
event_detection.py 의 결과(event CSV)를 읽어
"반복 패턴" 관점에서 이상 여부를 분석한다.

컨닝 핵심 패턴
--------------
① 같은 방향 고빈도 반복  : 오른쪽을 짧은 간격으로 N번 반복 보기
② 왔다갔다 교차 패턴     : 우→좌→우 또는 좌→우→좌 반복 (메모 ↔ 화면)
③ 버스트(burst)          : 짧은 시간 안에 같은 방향 집중 반복
④ 급속 재응시            : 직전 같은 방향 event 종료 후 매우 짧은 간격 재발

입력:  data/event_csv/_all_events.csv
출력:
  data/results/repetition_summary.csv   영상별 반복 패턴 요약
  data/results/burst_events.csv          버스트 구간 상세
  data/results/alternating_events.csv    왔다갔다 구간 상세
  콘솔 리포트 (의심도 랭킹 포함)

의심도 점수(suspicion_score) 계산 방식
----------------------------------------
  + 3점  per  같은 방향 event가 3회 이상인 방향 1개당
  + 2점  per  방향 전환(교차) 1회
  + 5점  per  버스트 1회
  + 4점  per  급속 재응시(간격 < RAPID_THRESHOLD) 1회
  → 총점이 높을수록 반복 패턴이 심함 (절댓값 기준 아님, 정상 분포와 비교 사용)

주요 파라미터 (경험적 기준값 — 정상 데이터 분포로 조정 가능)
------------------------------------------------------------
  BURST_WINDOW       = 10.0s  : 이 시간 안에 BURST_MIN_COUNT회 이상이면 burst
  BURST_MIN_COUNT    = 3       : burst 판정 최소 event 수
  RAPID_THRESHOLD    = 2.0s   : 이 간격 이하면 급속 재응시
  ALT_WINDOW         = 12.0s  : 이 시간 안에 교차 패턴 탐지
  ALT_MIN_SWITCHES   = 2       : 교차 판정 최소 방향 전환 수
"""

import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
import pandas as pd
import numpy as np
from itertools import groupby

# ══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════════════

ALL_EVENTS_CSV = Path("data/event_csv/_all_events.csv")
RESULT_DIR     = Path("data/results")

# ── 탐지 파라미터 (수정 근거 아래 설명) ────────────────────────────────────
#
# RAPID_THRESHOLD  1.5s  (이전 2.0s)
#   근거: 정상 데이터 재응시 간격 p25=0.7s, p50=1.4~2.0s
#         2.0s 기준은 정상 데이터의 절반을 급속 재응시로 잡아 너무 관대함
#         1.5s = p50 수준 → 중앙값보다 빠른 재응시만 급속으로 판정
#
# BURST_WINDOW / BURST_MIN_COUNT  8s / 4회  (이전 10s / 3회)
#   근거: 생각하다 자연스럽게 3번 같은 방향을 볼 수 있음
#         4번 이상이고 8초 내 집중된 경우 = 특정 외부 자료를 반복 확인하는 패턴
#
# ALT_WINDOW / ALT_MIN_SWITCHES  10s / 3회  (이전 12s / 2회)
#   근거: 2번 교차는 우연일 수 있음 (오른쪽 보고 생각하고 왼쪽 보기)
#         3번 이상 교차 = 화면↔메모 왔다갔다 독서 패턴
#
BURST_WINDOW      =  8.0   # seconds
BURST_MIN_COUNT   =  3     # 버스트 판정 최소 횟수
RAPID_THRESHOLD   =  1.5   # seconds — 이 이하 간격이면 급속 재응시
ALT_WINDOW        = 10.0   # seconds — 교차 패턴 탐지 창
ALT_MIN_SWITCHES  =  3     # 교차 방향 전환 최소 횟수

# ── 의심도 점수 가중치 ────────────────────────────────────────────────────────
#   높은 반복 횟수(4회+)와 버스트에 가중치를 더 줌
#   단순 교차보다 집중 반복이 더 의심스러운 신호이기 때문
W_HIGH_REPEAT  = 3   # 같은 방향 4회 이상인 방향 1개당
W_SWITCH       = 2   # 방향 전환 1회당
W_BURST        = 5   # 버스트 1회당
W_RAPID        = 4   # 급속 재응시 1회당

DIRECTIONS = ["right_reference", "left_reference", "down_reference"]
HL_DIRS    = ["right_reference", "left_reference"]  # 좌우(수평) 방향만 교차 탐지에 사용


# ══════════════════════════════════════════════════════════════════════════════
# 1. 데이터 로드
# ══════════════════════════════════════════════════════════════════════════════

def load_events(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"event 파일 없음: {path}\n"
            "  → python event_detection.py 를 먼저 실행하세요."
        )
    df = pd.read_csv(path, dtype={"video_id": str})
    df["start_time"]   = pd.to_numeric(df["start_time"],   errors="coerce")
    df["end_time"]     = pd.to_numeric(df["end_time"],     errors="coerce")
    df["duration_sec"] = pd.to_numeric(df["duration_sec"], errors="coerce")
    return df.dropna(subset=["start_time", "end_time"])


# ══════════════════════════════════════════════════════════════════════════════
# 2. 영상별 반복 통계
# ══════════════════════════════════════════════════════════════════════════════

def compute_direction_stats(events: pd.DataFrame) -> dict:
    """
    방향별 횟수, 간격 통계를 계산한다.
    - {dir}_count        : 해당 방향 event 총 횟수
    - {dir}_avg_interval : 연속 event 간 평균 간격 (초)
    - {dir}_min_interval : 연속 event 간 최소 간격 (초)
    - {dir}_rapid_count  : 급속 재응시 횟수 (간격 < RAPID_THRESHOLD)
    """
    stats = {}
    for d in DIRECTIONS:
        sub = events[events["event_type"] == d].sort_values("start_time")
        n   = len(sub)
        stats[f"{d}_count"] = n

        if n >= 2:
            # 직전 event 종료 시점 → 현재 event 시작 시점까지의 간격
            intervals = (sub["start_time"].values[1:] - sub["end_time"].values[:-1])
            intervals = np.maximum(intervals, 0.0)  # 음수 방지
            stats[f"{d}_avg_interval"] = round(float(np.mean(intervals)), 3)
            stats[f"{d}_min_interval"] = round(float(np.min(intervals)),  3)
            stats[f"{d}_rapid_count"]  = int(np.sum(intervals < RAPID_THRESHOLD))
        else:
            stats[f"{d}_avg_interval"] = None
            stats[f"{d}_min_interval"] = None
            stats[f"{d}_rapid_count"]  = 0

    return stats


# ══════════════════════════════════════════════════════════════════════════════
# 3. 교차(왔다갔다) 패턴 탐지
# ══════════════════════════════════════════════════════════════════════════════

def detect_alternating(events: pd.DataFrame, video_id: str) -> list[dict]:
    """
    좌우 event를 시간 순으로 나열해 방향 전환 횟수를 계산한다.
    ALT_WINDOW 내에서 ALT_MIN_SWITCHES 이상 교차하면 alternating event로 기록.

    예) right→left→right → 2번 전환 → 감지
    """
    hl = events[events["event_type"].isin(HL_DIRS)].sort_values("start_time")
    if len(hl) < ALT_MIN_SWITCHES + 1:
        return []

    records = []
    times  = hl["start_time"].values
    types  = hl["event_type"].values
    ends   = hl["end_time"].values

    for i in range(len(times)):
        # ALT_WINDOW 내의 이벤트 수집
        window_mask = (times >= times[i]) & (times <= times[i] + ALT_WINDOW)
        window_types = types[window_mask]
        window_times = times[window_mask]
        window_ends  = ends[window_mask]

        if len(window_types) < ALT_MIN_SWITCHES + 1:
            continue

        # 연속 방향 전환 횟수
        switches = sum(
            1 for a, b in zip(window_types[:-1], window_types[1:]) if a != b
        )

        if switches >= ALT_MIN_SWITCHES:
            records.append({
                "video_id":       video_id,
                "window_start":   round(float(window_times[0]),  3),
                "window_end":     round(float(window_ends[-1]),  3),
                "duration_sec":   round(float(window_ends[-1] - window_times[0]), 3),
                "event_count":    int(len(window_types)),
                "direction_switches": int(switches),
                "sequence":       "→".join(
                    t.replace("_reference","") for t in window_types
                ),
            })

    # 중복 창 제거: window_start가 같은 것은 switches가 가장 큰 것만 유지
    if not records:
        return []
    df_rec = pd.DataFrame(records)
    df_rec = (df_rec
              .sort_values("direction_switches", ascending=False)
              .drop_duplicates(subset=["window_start"])
              .sort_values("window_start")
              .reset_index(drop=True))
    return df_rec.to_dict(orient="records")


def summarize_alternating(alt_records: list[dict]) -> dict:
    if not alt_records:
        return {"alternating_windows": 0, "max_switches_in_window": 0, "total_direction_switches": 0}
    df = pd.DataFrame(alt_records)
    return {
        "alternating_windows":      len(df),
        "max_switches_in_window":   int(df["direction_switches"].max()),
        "total_direction_switches": int(df["direction_switches"].sum()),
    }


# ══════════════════════════════════════════════════════════════════════════════
# 4. 버스트 탐지
# ══════════════════════════════════════════════════════════════════════════════

def detect_bursts(events: pd.DataFrame, video_id: str) -> list[dict]:
    """
    각 방향별로 BURST_WINDOW 내에 BURST_MIN_COUNT 이상 event가 발생하면
    burst로 기록한다.
    """
    bursts = []
    for d in DIRECTIONS:
        sub = events[events["event_type"] == d].sort_values("start_time")
        if len(sub) < BURST_MIN_COUNT:
            continue

        times = sub["start_time"].values
        ends  = sub["end_time"].values

        for i in range(len(times)):
            # 슬라이딩 윈도우: times[i] 기준 BURST_WINDOW 내 events
            mask  = (times >= times[i]) & (times <= times[i] + BURST_WINDOW)
            count = int(np.sum(mask))
            if count >= BURST_MIN_COUNT:
                bursts.append({
                    "video_id":       video_id,
                    "burst_type":     d,
                    "event_count":    count,
                    "window_start":   round(float(times[i]),       3),
                    "window_end":     round(float(times[mask][-1] if mask.sum() > 0 else times[i] + BURST_WINDOW), 3),
                    "window_duration":round(float(min(BURST_WINDOW, ends[mask][-1] - times[i]) if mask.sum() > 0 else BURST_WINDOW), 3),
                })

    # 중복 창 제거 (같은 방향, 같은 window_start)
    if not bursts:
        return []
    df_b = pd.DataFrame(bursts)
    df_b = (df_b
            .sort_values("event_count", ascending=False)
            .drop_duplicates(subset=["burst_type", "window_start"])
            .sort_values(["burst_type", "window_start"])
            .reset_index(drop=True))
    return df_b.to_dict(orient="records")


def summarize_bursts(burst_records: list[dict]) -> dict:
    if not burst_records:
        return {"burst_count": 0, "burst_max_events": 0}
    df = pd.DataFrame(burst_records)
    return {
        "burst_count":      len(df),
        "burst_max_events": int(df["event_count"].max()),
    }


# ══════════════════════════════════════════════════════════════════════════════
# 5. 의심도 점수 계산
# ══════════════════════════════════════════════════════════════════════════════

def compute_suspicion_score(row: dict) -> int:
    """
    반복 패턴 기반 의심도 점수 (높을수록 반복 의심 패턴이 강함).

    항목별 가중치:
      같은 방향 3회 이상인 방향 1개당   + W_HIGH_REPEAT(3)
      방향 전환(교차) 1회당             + W_SWITCH(2)
      버스트 1회당                      + W_BURST(5)
      급속 재응시 1회당                 + W_RAPID(4)
    """
    score = 0
    for d in DIRECTIONS:
        cnt = row.get(f"{d}_count", 0) or 0
        if cnt >= 3:
            score += W_HIGH_REPEAT
        score += (row.get(f"{d}_rapid_count", 0) or 0) * W_RAPID

    score += (row.get("total_direction_switches", 0) or 0) * W_SWITCH
    score += (row.get("burst_count", 0) or 0) * W_BURST
    return score


# ══════════════════════════════════════════════════════════════════════════════
# 6. 전체 배치 처리
# ══════════════════════════════════════════════════════════════════════════════

def analyze_all(df_events: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    전체 영상을 처리하여 세 개의 결과 DataFrame을 반환한다.
    Returns: (summary_df, burst_df, alternating_df)
    """
    all_summary    = []
    all_bursts     = []
    all_alternating = []

    video_ids = sorted(df_events["video_id"].unique())
    print(f"  {len(video_ids)}개 영상 반복 패턴 분석 중...")

    for vid in video_ids:
        ev = df_events[df_events["video_id"] == vid].copy()

        # ── 방향별 통계 ──────────────────────────────────────────────────
        dir_stats = compute_direction_stats(ev)

        # ── 교차(왔다갔다) 탐지 ──────────────────────────────────────────
        alt_records = detect_alternating(ev, vid)
        alt_stats   = summarize_alternating(alt_records)
        all_alternating.extend(alt_records)

        # ── 버스트 탐지 ──────────────────────────────────────────────────
        burst_records = detect_bursts(ev, vid)
        burst_stats   = summarize_bursts(burst_records)
        all_bursts.extend(burst_records)

        # ── 전체 합산 ─────────────────────────────────────────────────────
        total_rapid = sum(
            dir_stats.get(f"{d}_rapid_count", 0) or 0 for d in DIRECTIONS
        )
        row = {
            "video_id":           vid,
            "total_events":       len(ev),
            **dir_stats,
            **alt_stats,
            **burst_stats,
            "total_rapid_repeats": total_rapid,
        }
        row["suspicion_score"] = compute_suspicion_score(row)
        all_summary.append(row)

    summary_df     = pd.DataFrame(all_summary)
    burst_df       = pd.DataFrame(all_bursts) if all_bursts else pd.DataFrame()
    alternating_df = pd.DataFrame(all_alternating) if all_alternating else pd.DataFrame()

    # 의심도 기준 정렬
    summary_df = summary_df.sort_values("suspicion_score", ascending=False).reset_index(drop=True)

    return summary_df, burst_df, alternating_df


# ══════════════════════════════════════════════════════════════════════════════
# 7. 콘솔 리포트
# ══════════════════════════════════════════════════════════════════════════════

def print_report(summary: pd.DataFrame,
                 burst_df: pd.DataFrame,
                 alt_df: pd.DataFrame) -> None:

    print()
    print("=" * 68)
    print("  반복 패턴 분석 결과 리포트")
    print("=" * 68)

    # ── 전체 통계 ─────────────────────────────────────────────────────────
    print(f"  분석 영상 수         : {len(summary)}")
    print(f"  의심 영상 (score>0)  : {(summary['suspicion_score'] > 0).sum()}")
    print(f"  버스트 감지 영상     : {(summary['burst_count'] > 0).sum()}")
    print(f"  교차 패턴 감지 영상  : {(summary['alternating_windows'] > 0).sum()}")
    print()

    # ── 방향별 반복 횟수 요약 ─────────────────────────────────────────────
    print("  [방향별 반복 횟수 분포 (전체 영상)]")
    for d in DIRECTIONS:
        col = f"{d}_count"
        label = d.replace("_reference", "")
        vals  = summary[col].fillna(0)
        print(f"    {label:<6}  mean={vals.mean():.2f}  max={int(vals.max())}  "
              f"3회이상={int((vals >= 3).sum())}개 영상  5회이상={int((vals >= 5).sum())}개 영상")
    print()

    # ── 의심도 상위 20개 ──────────────────────────────────────────────────
    cols_show = [
        "video_id",
        "right_reference_count", "left_reference_count", "down_reference_count",
        "total_direction_switches", "burst_count", "total_rapid_repeats",
        "suspicion_score",
    ]
    cols_show = [c for c in cols_show if c in summary.columns]
    print("  [의심도 상위 20개 영상]")
    top20 = summary.head(20)[cols_show].rename(columns={
        "right_reference_count": "right",
        "left_reference_count":  "left",
        "down_reference_count":  "down",
        "total_direction_switches": "switches",
        "burst_count":           "bursts",
        "total_rapid_repeats":   "rapids",
    })
    print(top20.to_string(index=False))
    print()

    # ── 버스트 상세 ───────────────────────────────────────────────────────
    if not burst_df.empty:
        print("  [감지된 버스트 구간 전체]")
        b_show = burst_df[["video_id","burst_type","event_count","window_start","window_end","window_duration"]]
        print(b_show.to_string(index=False))
        print()

    # ── 교차 패턴 상세 ────────────────────────────────────────────────────
    if not alt_df.empty:
        print("  [감지된 교차(왔다갔다) 패턴 전체]")
        a_show = alt_df[["video_id","window_start","window_end","direction_switches","event_count","sequence"]]
        a_show = a_show.sort_values("direction_switches", ascending=False)
        print(a_show.to_string(index=False))
        print()

    # ── 의심도 0 (완전 정상) 영상 수 ─────────────────────────────────────
    zero_score = (summary["suspicion_score"] == 0).sum()
    print(f"  의심도 0 (반복 패턴 없음) 영상: {zero_score}개")
    print("=" * 68)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    print("\n[1/3] event 데이터 로드")
    df_events = load_events(ALL_EVENTS_CSV)
    print(f"  총 event 수: {len(df_events)}  |  영상 수: {df_events['video_id'].nunique()}")

    print("\n[2/3] 반복 패턴 분석")
    summary, burst_df, alt_df = analyze_all(df_events)

    print("\n[3/3] 결과 저장")
    summary_path = RESULT_DIR / "repetition_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"  영상별 요약    → {summary_path}")

    if not burst_df.empty:
        burst_path = RESULT_DIR / "burst_events.csv"
        burst_df.to_csv(burst_path, index=False)
        print(f"  버스트 상세    → {burst_path}")

    if not alt_df.empty:
        alt_path = RESULT_DIR / "alternating_events.csv"
        alt_df.to_csv(alt_path, index=False)
        print(f"  교차 패턴 상세 → {alt_path}")

    print_report(summary, burst_df, alt_df)

    # 정상 데이터 기준선 출력 (추후 검증 영상 비교용)
    print("\n  [정상 데이터 기준선 — 검증 영상과 비교 기준으로 활용]")
    print(f"  suspicion_score 평균: {summary['suspicion_score'].mean():.2f}")
    print(f"  suspicion_score p75 : {summary['suspicion_score'].quantile(0.75):.1f}")
    print(f"  suspicion_score p95 : {summary['suspicion_score'].quantile(0.95):.1f}")
    print(f"  suspicion_score max : {summary['suspicion_score'].max():.0f}")
    print()
    print("  ※ 검증 영상의 score가 정상 p95 초과 시 → 반복 패턴 이상 의심")

    # 기준선 저장
    baseline = {
        "suspicion_mean": round(summary["suspicion_score"].mean(), 3),
        "suspicion_p75":  round(float(summary["suspicion_score"].quantile(0.75)), 3),
        "suspicion_p95":  round(float(summary["suspicion_score"].quantile(0.95)), 3),
        "suspicion_max":  int(summary["suspicion_score"].max()),
        "rapid_threshold_sec": RAPID_THRESHOLD,
        "burst_window_sec":    BURST_WINDOW,
        "burst_min_count":     BURST_MIN_COUNT,
        "alt_window_sec":      ALT_WINDOW,
        "alt_min_switches":    ALT_MIN_SWITCHES,
    }
    pd.DataFrame([baseline]).to_csv(RESULT_DIR / "repetition_baseline.csv", index=False)
    print(f"  기준선 저장    → {RESULT_DIR / 'repetition_baseline.csv'}")


# ══════════════════════════════════════════════════════════════════════════════
# 단일 영상 분석 래퍼 (final_pipeline.py 에서 호출)
# ══════════════════════════════════════════════════════════════════════════════

def analyze_video_repetition(events_df: pd.DataFrame, video_id: str) -> dict:
    """
    단일 영상의 이벤트 DataFrame을 받아 반복 패턴 분석 결과를 dict로 반환.
    final_pipeline.py 에서 사용.
    """
    if events_df.empty:
        return {"right": 0, "left": 0, "down": 0,
                "switches": 0, "bursts": 0, "rapids": 0, "suspicion_score": 0}

    dir_stats     = compute_direction_stats(events_df)
    alt_records   = detect_alternating(events_df, video_id)
    alt_stats     = summarize_alternating(alt_records)
    burst_records = detect_bursts(events_df, video_id)
    burst_stats   = summarize_bursts(burst_records)

    total_rapid = sum(
        dir_stats.get(f"{d}_rapid_count", 0) or 0 for d in DIRECTIONS
    )
    row = {
        "video_id":            video_id,
        "total_events":        len(events_df),
        **dir_stats,
        **alt_stats,
        **burst_stats,
        "total_rapid_repeats": total_rapid,
    }
    row["suspicion_score"] = compute_suspicion_score(row)

    return {
        "right":           dir_stats.get("right_count",  0),
        "left":            dir_stats.get("left_count",   0),
        "down":            dir_stats.get("down_count",   0),
        "switches":        alt_stats.get("total_switches", 0),
        "bursts":          burst_stats.get("total_bursts",  0),
        "rapids":          total_rapid,
        "suspicion_score": row["suspicion_score"],
    }


if __name__ == "__main__":
    main()
