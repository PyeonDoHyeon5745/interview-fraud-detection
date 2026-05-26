#!/usr/bin/env python3
"""
evaluate_results.py
===================
테스트 영상 분석 결과를 정량화하여 평가한다.

출력:
  - 전체 시스템 혼동행렬 + Precision / Recall / F1
  - 모듈별 (visual / audio / yolo) 감지 성능
  - 이벤트 타입별 통계 (건수, 평균 지속시간)
  - 결과 → data/results/test/evaluation_report.csv

사용법:
  python evaluate_results.py
"""
from pathlib import Path
import pandas as pd
import numpy as np

RESULT_DIR = Path("data/results/test")

# ── Ground Truth ──────────────────────────────────────────────────────────────
# 각 영상의 정답 레이블 (전체 + 모듈별)
# label=1: 부정행위 있음, label=0: 정상
GROUND_TRUTH = {
    #  video_id             overall  visual  audio  yolo
    "1 정상":          {"overall": 0, "visual": 0, "audio": 0, "yolo": 0},
    "2 좌+우 반복":    {"overall": 1, "visual": 1, "audio": 0, "yolo": 0},
    "3 우측 오래":     {"overall": 1, "visual": 1, "audio": 0, "yolo": 0},
    "4 버스트":        {"overall": 1, "visual": 1, "audio": 0, "yolo": 0},
    "5 하향":          {"overall": 1, "visual": 1, "audio": 0, "yolo": 0},
    "6 눈동자만":      {"overall": 1, "visual": 1, "audio": 0, "yolo": 0},
    "7 외부음성":      {"overall": 1, "visual": 0, "audio": 1, "yolo": 0},
    "8 물건탐지":      {"overall": 1, "visual": 0, "audio": 0, "yolo": 1},
}

# ── 유틸 ──────────────────────────────────────────────────────────────────────
def metrics(tp, fp, fn, tn):
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy  = (tp + tn) / (tp + fp + fn + tn) if (tp + fp + fn + tn) > 0 else 0.0
    return precision, recall, f1, accuracy


def print_section(title):
    print(f"\n{'═'*62}")
    print(f"  {title}")
    print(f"{'═'*62}")


# ── 결과 로드 ─────────────────────────────────────────────────────────────────
def load_results():
    rows = []
    for video_id in GROUND_TRUTH:
        csv = RESULT_DIR / f"{video_id}_result.csv"
        if csv.exists():
            df = pd.read_csv(csv)
            df["video_id"] = video_id
            rows.append(df)
    if not rows:
        print("[ERROR] 결과 CSV 없음. run_test_videos.py 먼저 실행.")
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def main():
    df = load_results()
    if df.empty:
        return

    # ── 모듈별 감지 여부 집계 ────────────────────────────────────────────────
    detected = {}
    for video_id in GROUND_TRUTH:
        sub = df[df["video_id"] == video_id]
        detected[video_id] = {
            "visual": int(len(sub[sub["module"] == "visual"]) > 0),
            "audio":  int(len(sub[sub["module"] == "audio"])  > 0),
            "yolo":   int(len(sub[sub["module"] == "yolo"])   > 0),
        }
        detected[video_id]["overall"] = int(
            detected[video_id]["visual"] or
            detected[video_id]["audio"]  or
            detected[video_id]["yolo"]
        )

    # ── 전체 시스템 성능 ──────────────────────────────────────────────────────
    print_section("1. 전체 시스템 성능 (부정행위 감지)")

    records = []
    tp = fp = fn = tn = 0
    print(f"\n  {'영상':<22} {'정답':>6} {'예측':>6}  판정")
    print(f"  {'─'*48}")
    for vid, gt in GROUND_TRUTH.items():
        pred = detected[vid]["overall"]
        g    = gt["overall"]
        if   g == 1 and pred == 1: tp += 1; mark = "TP ✓"
        elif g == 0 and pred == 0: tn += 1; mark = "TN ✓"
        elif g == 0 and pred == 1: fp += 1; mark = "FP ✗"
        else:                      fn += 1; mark = "FN ✗"
        records.append({"video_id": vid, "gt_overall": g, "pred_overall": pred, "mark": mark})
        print(f"  {vid:<22} {'부정' if g else '정상':>6} {'부정' if pred else '정상':>6}  {mark}")

    print(f"\n  혼동행렬")
    print(f"  {'':15} 예측:정상  예측:부정")
    print(f"  {'정답:정상':<15}   TN={tn:2d}      FP={fp:2d}")
    print(f"  {'정답:부정':<15}   FN={fn:2d}      TP={tp:2d}")

    p, r, f1, acc = metrics(tp, fp, fn, tn)
    print(f"\n  Precision : {p:.4f}")
    print(f"  Recall    : {r:.4f}")
    print(f"  F1-Score  : {f1:.4f}")
    print(f"  Accuracy  : {acc:.4f}  ({tp+tn}/{tp+fp+fn+tn})")

    # ── 모듈별 성능 ───────────────────────────────────────────────────────────
    print_section("2. 모듈별 감지 성능")

    module_stats = []
    for module in ["visual", "audio", "yolo"]:
        mtp = mfp = mfn = mtn = 0
        for vid, gt in GROUND_TRUTH.items():
            g    = gt[module]
            pred = detected[vid][module]
            if   g == 1 and pred == 1: mtp += 1
            elif g == 0 and pred == 0: mtn += 1
            elif g == 0 and pred == 1: mfp += 1
            else:                      mfn += 1
        mp, mr, mf1, macc = metrics(mtp, mfp, mfn, mtn)
        module_stats.append({
            "module": module, "TP": mtp, "FP": mfp, "FN": mfn, "TN": mtn,
            "Precision": mp, "Recall": mr, "F1": mf1, "Accuracy": macc
        })
        print(f"\n  [{module.upper()}]")
        print(f"  TP={mtp}  FP={mfp}  FN={mfn}  TN={mtn}")
        print(f"  Precision={mp:.4f}  Recall={mr:.4f}  F1={mf1:.4f}  Accuracy={macc:.4f}")

    # ── 이벤트 타입별 통계 ───────────────────────────────────────────────────
    print_section("3. 이벤트 타입별 통계")

    if "duration" in df.columns and "event_type" in df.columns:
        grp = df.groupby(["module", "event_type"])["duration"].agg(
            count="count", mean="mean", std="std", min="min", max="max"
        ).reset_index()
        print(f"\n  {'모듈':<8} {'이벤트 타입':<25} {'건수':>5} {'평균(s)':>8} {'최소(s)':>8} {'최대(s)':>8}")
        print(f"  {'─'*66}")
        for _, row in grp.iterrows():
            std_str = f"±{row['std']:.2f}" if not pd.isna(row['std']) else "     "
            print(f"  {row['module']:<8} {row['event_type']:<25} {row['count']:>5} "
                  f"  {row['mean']:>5.2f}{std_str}  {row['min']:>5.2f}    {row['max']:>5.2f}")

    # ── 음성 유사도 통계 ─────────────────────────────────────────────────────
    print_section("4. 음성 이벤트 상세")

    audio_rows = []
    for video_id in GROUND_TRUTH:
        csv = RESULT_DIR / f"{video_id}_result.csv"
        if csv.exists():
            sub = pd.read_csv(csv)
            sub = sub[sub["module"] == "audio"]
            if not sub.empty:
                sub["video_id"] = video_id
                audio_rows.append(sub)

    if audio_rows:
        adf = pd.concat(audio_rows, ignore_index=True)
        print(f"\n  {'영상':<22} {'시작(s)':>8} {'종료(s)':>8} {'지속(s)':>8}")
        print(f"  {'─'*54}")
        for _, row in adf.iterrows():
            print(f"  {row['video_id']:<22} {row['start']:>8.1f} {row['end']:>8.1f} {row['duration']:>8.1f}")
    else:
        print("\n  (음성 이벤트 없음)")

    # ── YOLO 이벤트 상세 ─────────────────────────────────────────────────────
    print_section("5. YOLO 객체 탐지 상세")

    yolo_rows = []
    for video_id in GROUND_TRUTH:
        csv = RESULT_DIR / f"{video_id}_result.csv"
        if csv.exists():
            sub = pd.read_csv(csv)
            sub = sub[sub["module"] == "yolo"]
            if not sub.empty:
                sub["video_id"] = video_id
                yolo_rows.append(sub)

    if yolo_rows:
        ydf = pd.concat(yolo_rows, ignore_index=True)
        print(f"\n  {'영상':<22} {'객체':<15} {'시작(s)':>8} {'종료(s)':>8} {'지속(s)':>8}")
        print(f"  {'─'*64}")
        for _, row in ydf.iterrows():
            print(f"  {row['video_id']:<22} {row['event_type']:<15} {row['start']:>8.1f} {row['end']:>8.1f} {row['duration']:>8.1f}")
    else:
        print("\n  (YOLO 이벤트 없음)")

    # ── CSV 저장 ─────────────────────────────────────────────────────────────
    report_rows = []
    for r in records:
        vid = r["video_id"]
        row = {"video_id": vid, "gt_overall": r["gt_overall"], "pred_overall": r["pred_overall"]}
        for m in ["visual", "audio", "yolo"]:
            row[f"gt_{m}"]   = GROUND_TRUTH[vid][m]
            row[f"pred_{m}"] = detected[vid][m]
        report_rows.append(row)

    report_df = pd.DataFrame(report_rows)
    out_path  = RESULT_DIR / "evaluation_report.csv"
    report_df.to_csv(out_path, index=False)

    ms_df = pd.DataFrame(module_stats)
    ms_df.to_csv(RESULT_DIR / "module_metrics.csv", index=False)

    print(f"\n\n  저장 완료")
    print(f"  → {out_path}")
    print(f"  → {RESULT_DIR / 'module_metrics.csv'}")
    print(f"{'═'*62}\n")


if __name__ == "__main__":
    main()
