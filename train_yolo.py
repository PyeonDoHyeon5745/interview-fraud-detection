#!/usr/bin/env python3
"""
train_yolo.py
=============
YOLOv11n 을 커스텀 4-클래스 데이터셋으로 파인튜닝한다.

클래스:
  0: book    1: phone    2: tablet    3: laptop

사전 조건:
  python setup_tablet_dataset.py 를 먼저 실행해 데이터셋을 준비할 것.

출력:
  data/yolo_weights/best.pt   ← 최종 사용할 가중치
"""
from pathlib import Path
from ultralytics import YOLO

# ─────────────────────────────────────────────────────────────────
# 설정
# ─────────────────────────────────────────────────────────────────
DATASET_YAML = Path("data/yolo_dataset/dataset.yaml")
OUTPUT_DIR   = Path("data/yolo_weights")

# 학습 하이퍼파라미터
EPOCHS      = 50       # 에폭 (GPU 있으면 100 권장)
IMG_SIZE    = 640      # 입력 이미지 크기
BATCH_SIZE  = 16       # 배치 크기 (메모리 부족 시 8로 줄이기)
PATIENCE    = 15       # Early stopping patience
DEVICE      = "mps"    # Mac M-chip: "mps" / NVIDIA GPU: 0 / CPU: "cpu"

# ─────────────────────────────────────────────────────────────────
def main():
    if not DATASET_YAML.exists():
        raise FileNotFoundError(
            f"데이터셋 없음: {DATASET_YAML}\n"
            "  → python setup_tablet_dataset.py 를 먼저 실행하세요."
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  YOLOv11 커스텀 학습 시작")
    print(f"  데이터셋: {DATASET_YAML}")
    print(f"  에폭: {EPOCHS}  배치: {BATCH_SIZE}  디바이스: {DEVICE}")
    print("=" * 60)

    # COCO pretrained YOLOv11n 로드 (book/phone/laptop 이미 학습됨)
    model = YOLO("yolo11n.pt")

    # 파인튜닝 — COCO 가중치에서 시작해 4개 클래스로 재학습
    results = model.train(
        data     = str(DATASET_YAML.resolve()),
        epochs   = EPOCHS,
        imgsz    = IMG_SIZE,
        batch    = BATCH_SIZE,
        patience = PATIENCE,
        device   = DEVICE,
        project  = str(OUTPUT_DIR),
        name     = "interview_detector",
        exist_ok = True,

        # 데이터 증강 (면접 환경 다양성 반영)
        hsv_h    = 0.015,   # 색조
        hsv_s    = 0.7,     # 채도
        hsv_v    = 0.4,     # 밝기
        flipud   = 0.0,     # 상하 반전 OFF (면접 영상에서 불필요)
        fliplr   = 0.5,     # 좌우 반전
        mosaic   = 1.0,     # 모자이크 증강
        scale    = 0.5,     # 스케일 변화

        # 로깅
        verbose  = True,
        plots    = True,    # 학습 곡선 저장
    )

    # 최적 가중치 복사
    best_pt = Path(results.save_dir) / "weights" / "best.pt"
    final_pt = OUTPUT_DIR / "best.pt"
    if best_pt.exists():
        import shutil
        shutil.copy2(best_pt, final_pt)
        print(f"\n최적 가중치 저장 → {final_pt.resolve()}")
    else:
        print(f"\n[WARN] best.pt 를 찾을 수 없습니다: {best_pt}")

    # 검증 성능 출력
    print("\n[검증 결과]")
    metrics = model.val()
    print(f"  mAP50    : {metrics.box.map50:.4f}")
    print(f"  mAP50-95 : {metrics.box.map:.4f}")

    print("\n완료. 다음 단계: python final_pipeline.py")


if __name__ == "__main__":
    main()
