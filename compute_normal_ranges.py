#!/usr/bin/env python3
"""
compute_normal_ranges.py
========================
정상 면접 데이터(data/frame_csv/*.csv)를 읽어
각 feature의 통계적 기준선(정상 범위)을 계산하고 저장한다.

출력 1: data/model/normal_ranges.csv
  컬럼: feature, mean, std, min, max, p05, p25, p50, p75, p95, iqr,
         thr_lo, thr_hi,
         thr_delta   ← 영상별 중앙값 보정 후 사용하는 편차 임계값 (2.0 × IQR)

출력 2: data/model/per_video_baseline.csv
  컬럼: video_id, median_yaw, median_pitch, median_eye_x_ratio, median_eye_y_ratio
  → 각 영상의 특성값 중앙값 (카메라 각도 개인차 보정용 기준선)

사용 feature:
  head_yaw, head_pitch, head_roll,
  eye_x_ratio, eye_y_ratio, eye_open

──────────────────────────────────────────────────
per-video baseline 설계 근거:
  영상마다 카메라 위치·각도가 달라 head_yaw 절대값에
  수 도 ~ 십수 도의 오프셋이 존재한다.
  각 영상의 중앙값(median)을 개인 기준선으로 사용하면
  이 구조적 오프셋을 제거하고 '본인 평소 대비 얼마나 돌렸는가'
  를 측정할 수 있다.
  median은 절사 평균 중 breakdown point가 50%로 가장 높아
  이상 프레임이 전체의 50% 미만이면 기준선이 왜곡되지 않는다.
──────────────────────────────────────────────────
"""

import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
import pandas as pd
import numpy as np

# ══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════════════

FRAME_CSV_DIR    = Path("data/frame_csv")
OUTPUT_DIR       = Path("data/model")
OUTPUT_FILE      = OUTPUT_DIR / "normal_ranges.csv"
BASELINE_FILE    = OUTPUT_DIR / "per_video_baseline.csv"

# per-video baseline에서 추적할 feature
BASELINE_FEATS = ["head_yaw", "head_pitch", "eye_x_ratio", "eye_y_ratio"]

# per-video delta flag에 사용할 IQR 배수
DELTA_K = 2.0

FEATURE_COLS = [
    "head_yaw",
    "head_pitch",
    "head_roll",
    "eye_x_ratio",
    "eye_y_ratio",
    "eye_open",
]


# ══════════════════════════════════════════════════════════════════════════════
# 데이터 로드
# ══════════════════════════════════════════════════════════════════════════════

def load_all_csv(csv_dir: Path) -> pd.DataFrame:
    """data/frame_csv/ 내 모든 CSV를 하나의 DataFrame으로 합친다."""
    files = sorted(csv_dir.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"CSV 파일 없음: {csv_dir.resolve()}")

    print(f"  {len(files)}개 CSV 로드 중...")
    dfs = []
    for f in files:
        try:
            dfs.append(pd.read_csv(f, dtype={"video_id": str}))
        except Exception as e:
            print(f"  [WARN] {f.name}: {e}")

    df = pd.concat(dfs, ignore_index=True)
    print(f"  전체 프레임: {len(df):,}")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# 전처리
# ══════════════════════════════════════════════════════════════════════════════

def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """
    1. face_detected == 1 인 프레임만 사용
    2. feature 컬럼을 float로 변환
    3. NaN 행 제거
    """
    df_face = df[df["face_detected"] == 1].copy()
    print(f"  얼굴 검출 프레임: {len(df_face):,} / {len(df):,}")

    for col in FEATURE_COLS:
        df_face[col] = pd.to_numeric(df_face[col], errors="coerce")

    df_valid = df_face.dropna(subset=FEATURE_COLS).copy()
    print(f"  유효 프레임 (NaN 제거 후): {len(df_valid):,}")
    return df_valid


# ══════════════════════════════════════════════════════════════════════════════
# 통계 계산
# ══════════════════════════════════════════════════════════════════════════════

def compute_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    각 feature에 대해 분포 통계를 계산한다.

    반환 컬럼:
        feature, mean, std, min, max,
        p025, p05, p25, p50, p75, p95, p975, iqr,
        thr_lo, thr_hi   ← event_detection.py 에서 사용하는 실제 threshold

    threshold 선택 근거
    -------------------
    head_yaw / head_pitch (각도, 대략 Gaussian 분포):
        thr_lo = mean - 2.5 * std
        thr_hi = mean + 2.5 * std
        → p05/p95(상하위5%) 대신 ±2.5σ(상하위0.6%) 로 넓힘
        → 생각하다 자연스럽게 고개가 돌아가는 정도(≈±30°)는 허용

    eye_x_ratio / eye_y_ratio (비율, 이상치 존재):
        thr_lo = p025 (2.5 백분위)
        thr_hi = p975 (97.5 백분위)
        → 이상치가 σ를 왜곡하므로 분위수 기반 사용
        → p05/p95보다 2배 넓은 범위 허용

    thr_delta (per-video baseline 보정 후 사용):
        thr_delta = DELTA_K × IQR  (= 2.0 × IQR)
        → 각 영상의 median을 빼고 |delta| > thr_delta 이면 flag
        → IQR은 이상치에 강건한 산포 지표 (breakdown point 25%)
    """
    # head 각도 feature: Gaussian 가정으로 ±2.5σ
    HEAD_FEATS = {"head_yaw", "head_pitch", "head_roll"}

    rows = []
    for feat in FEATURE_COLS:
        s   = df[feat].dropna()
        mu  = float(s.mean())
        sig = float(s.std())
        p25 = float(s.quantile(0.25))
        p75 = float(s.quantile(0.75))
        p025 = float(s.quantile(0.025))
        p975 = float(s.quantile(0.975))

        if feat in HEAD_FEATS:
            thr_lo = mu - 2.5 * sig
            thr_hi = mu + 2.5 * sig
        else:
            thr_lo = p025
            thr_hi = p975

        iqr_val = p75 - p25
        rows.append({
            "feature":   feat,
            "mean":      round(mu,  6),
            "std":       round(sig, 6),
            "min":       round(float(s.min()), 6),
            "max":       round(float(s.max()), 6),
            "p025":      round(p025, 6),
            "p05":       round(float(s.quantile(0.05)), 6),
            "p25":       round(p25,  6),
            "p50":       round(float(s.quantile(0.50)), 6),
            "p75":       round(p75,  6),
            "p95":       round(float(s.quantile(0.95)), 6),
            "p975":      round(p975, 6),
            "iqr":       round(iqr_val, 6),
            "thr_lo":    round(thr_lo, 6),
            "thr_hi":    round(thr_hi, 6),
            # per-video delta threshold: 영상별 median 보정 후 |delta| > thr_delta 이면 flag
            # BASELINE_FEATS에 속하지 않는 feature는 NaN
            "thr_delta": round(DELTA_K * iqr_val, 6) if feat in BASELINE_FEATS else float("nan"),
        })
    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════════════════════
# per-video baseline 계산
# ══════════════════════════════════════════════════════════════════════════════

def compute_per_video_baseline(csv_dir: Path) -> pd.DataFrame:
    """
    각 영상의 BASELINE_FEATS에 대한 중앙값(median)을 계산한다.

    반환 컬럼:
        video_id, median_yaw, median_pitch, median_eye_x_ratio, median_eye_y_ratio

    설계 근거:
        median은 breakdown point가 50%로, 해당 영상의 프레임 절반 이상이
        이상 프레임이 아닌 한 기준선이 왜곡되지 않는다.
        영상별 중앙값 = 해당 응시자의 '정상 정면 응시 기준점'으로 해석된다.
        이를 빼고 편차(delta)만 분석하면 카메라 각도·개인차를 모두 제거한
        '본인 평소 대비 이탈 정도'를 측정할 수 있다.
    """
    files = sorted(csv_dir.glob("*.csv"))
    rows  = []

    for f in files:
        try:
            df = pd.read_csv(f, dtype={"video_id": str})
        except Exception as e:
            print(f"  [WARN] {f.name}: {e}")
            continue

        # face_detected == 1인 프레임만
        df_face = df[df["face_detected"] == 1].copy()
        if df_face.empty:
            continue

        for col in BASELINE_FEATS:
            if col in df_face.columns:
                df_face[col] = pd.to_numeric(df_face[col], errors="coerce")

        df_valid = df_face.dropna(subset=BASELINE_FEATS)
        if df_valid.empty:
            continue

        video_id = f.stem
        row = {"video_id": video_id}
        col_map = {
            "head_yaw":      "median_yaw",
            "head_pitch":    "median_pitch",
            "eye_x_ratio":   "median_eye_x_ratio",
            "eye_y_ratio":   "median_eye_y_ratio",
        }
        for feat, col_name in col_map.items():
            row[col_name] = round(float(df_valid[feat].median()), 6)

        rows.append(row)

    if not rows:
        raise RuntimeError("per-video baseline 계산 실패: 유효 영상 없음")

    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("\n[1/4] 데이터 로드")
    df_all = load_all_csv(FRAME_CSV_DIR)

    print("\n[2/4] 전처리")
    df_valid = preprocess(df_all)

    print("\n[3/4] 전체 통계 계산 (normal_ranges)")
    stats = compute_stats(df_valid)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    stats.to_csv(OUTPUT_FILE, index=False)
    print(f"  저장 완료 → {OUTPUT_FILE.resolve()}")
    print()
    print(stats[["feature","mean","std","p05","p95","thr_lo","thr_hi","thr_delta"]].to_string(index=False))

    print("\n[4/4] 영상별 중앙값 기준선 계산 (per_video_baseline)")
    baseline = compute_per_video_baseline(FRAME_CSV_DIR)
    baseline.to_csv(BASELINE_FILE, index=False)
    print(f"  저장 완료 → {BASELINE_FILE.resolve()}")
    print(f"  영상 수: {len(baseline)}")
    print()

    # ── 요약 출력 ──────────────────────────────────────────────────────────
    nr = stats.set_index("feature")
    print("=" * 72)
    print("  [A] 절대 threshold (thr_lo / thr_hi) — 전체 집단 기준")
    print("  ※ head 각도: mean±2.5σ  /  eye 비율: p2.5~p97.5")
    print("=" * 72)
    rows_desc = [
        ("head_yaw",    "좌측 편향",            "우측 편향"),
        ("head_pitch",  "위 편향(뒤로 젖힘)",   "아래 편향(고개 숙임)"),
        ("eye_x_ratio", "홍채 좌측",             "홍채 우측"),
        ("eye_y_ratio", "홍채 위(위 쳐다보기)",  "홍채 아래(아래 보기)"),
    ]
    for feat, note_lo, note_hi in rows_desc:
        lo = nr.loc[feat, "thr_lo"]
        hi = nr.loc[feat, "thr_hi"]
        print(f"  {feat:<15} thr_lo={lo:>8.4f}  thr_hi={hi:>8.4f}")
        print(f"    < thr_lo → {note_lo}")
        print(f"    > thr_hi → {note_hi}")

    print()
    print("=" * 72)
    print("  [B] per-video delta threshold (thr_delta = 2.0 × IQR) — 카메라 보정 기준")
    print("  사용 방법: delta = feature - video_median,  flag if |delta| > thr_delta")
    print("=" * 72)
    for feat in BASELINE_FEATS:
        td = nr.loc[feat, "thr_delta"]
        iqr_v = nr.loc[feat, "iqr"]
        print(f"  {feat:<15} IQR={iqr_v:>7.4f}  thr_delta=±{td:.4f}")
    print("=" * 72)

    print()
    print("  [per-video baseline 통계]")
    print(baseline.describe().round(4).to_string())
    print()


if __name__ == "__main__":
    main()
