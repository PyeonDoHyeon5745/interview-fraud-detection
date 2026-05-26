#!/usr/bin/env python3
"""
setup_tablet_dataset.py
=======================
Open Images V7 에서 4개 클래스 이미지를 다운로드하고
YOLOv11 학습용 데이터셋 형식으로 변환한다.

클래스:
  0: book       (책)
  1: phone      (핸드폰)
  2: tablet     (태블릿) ← COCO에 없어서 커스텀 학습 필요
  3: laptop     (노트북)

출력 구조:
  data/yolo_dataset/
  ├── images/
  │   ├── train/
  │   └── val/
  ├── labels/
  │   ├── train/
  │   └── val/
  └── dataset.yaml
"""
import os
import shutil
import random
from pathlib import Path

import fiftyone as fo
import fiftyone.zoo as foz

# ─────────────────────────────────────────────────────────────────
# 설정
# ─────────────────────────────────────────────────────────────────
OUTPUT_DIR  = Path("data/yolo_dataset")
SAMPLES_PER_CLASS = 300       # 클래스당 다운로드 이미지 수
VAL_RATIO   = 0.15            # 검증 비율

# Open Images 클래스명 → 우리 클래스 레이블
CLASS_MAP = {
    "Book":             (0, "book"),
    "Mobile phone":     (1, "phone"),
    "Tablet computer":  (2, "tablet"),
    "Laptop":           (3, "laptop"),
}

# ─────────────────────────────────────────────────────────────────
def download_open_images():
    """Open Images V7 에서 4개 클래스 다운로드"""
    print("\n[1/3] Open Images 다운로드 중...")
    print(f"  클래스: {list(CLASS_MAP.keys())}")
    print(f"  클래스당 {SAMPLES_PER_CLASS}장 (train+val 합산)")

    datasets = {}
    for split in ["train", "validation"]:
        n = SAMPLES_PER_CLASS if split == "train" else max(50, SAMPLES_PER_CLASS // 5)
        print(f"\n  [{split}] {n}장 다운로드...")
        try:
            ds = foz.load_zoo_dataset(
                "open-images-v7",
                split=split,
                label_types=["detections"],
                classes=list(CLASS_MAP.keys()),
                max_samples=n,
                only_matching=True,
                dataset_name=f"oi_{split}_{n}",
                overwrite=True,
            )
            datasets[split] = ds
            print(f"  [{split}] {len(ds)}개 샘플 로드 완료")
        except Exception as e:
            print(f"  [WARN] {split} 다운로드 오류: {e}")
            datasets[split] = None

    return datasets


def convert_to_yolo(datasets: dict):
    """fiftyone 데이터셋 → YOLO 형식 변환"""
    print("\n[2/3] YOLO 형식 변환 중...")

    for split_name, ds in datasets.items():
        if ds is None:
            continue

        yolo_split = "train" if split_name == "train" else "val"
        img_dir = OUTPUT_DIR / "images" / yolo_split
        lbl_dir = OUTPUT_DIR / "labels" / yolo_split
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)

        count = 0
        for sample in ds:
            if not sample.filepath or not Path(sample.filepath).exists():
                continue

            detections = sample.ground_truth
            if detections is None or not detections.detections:
                continue

            # 유효한 클래스 bbox만 필터
            valid_lines = []
            for det in detections.detections:
                label = det.label
                if label not in CLASS_MAP:
                    continue
                class_id = CLASS_MAP[label][0]

                # fiftyone bbox: [x, y, w, h] (0~1 normalized, top-left)
                x, y, w, h = det.bounding_box
                cx = x + w / 2
                cy = y + h / 2
                # 경계 클리핑
                cx = max(0, min(1, cx))
                cy = max(0, min(1, cy))
                w  = max(0, min(1, w))
                h  = max(0, min(1, h))
                if w < 0.01 or h < 0.01:
                    continue
                valid_lines.append(f"{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

            if not valid_lines:
                continue

            # 이미지 복사
            src = Path(sample.filepath)
            stem = src.stem
            dst_img = img_dir / src.name
            shutil.copy2(src, dst_img)

            # 라벨 저장
            lbl_path = lbl_dir / f"{stem}.txt"
            with open(lbl_path, "w") as f:
                f.write("\n".join(valid_lines))

            count += 1

        print(f"  [{yolo_split}] {count}개 샘플 변환 완료")


def create_yaml():
    """dataset.yaml 생성"""
    yaml_path = OUTPUT_DIR / "dataset.yaml"
    yaml_content = f"""# YOLOv11 학습용 데이터셋 설정
# 클래스: 비대면 면접 부정행위 관련 객체

path: {OUTPUT_DIR.resolve()}
train: images/train
val:   images/val

nc: 4
names:
  0: book
  1: phone
  2: tablet
  3: laptop
"""
    with open(yaml_path, "w") as f:
        f.write(yaml_content)
    print(f"\n  dataset.yaml 저장 → {yaml_path.resolve()}")


def print_stats():
    """데이터셋 통계 출력"""
    print("\n[데이터셋 통계]")
    for split in ["train", "val"]:
        img_dir = OUTPUT_DIR / "images" / split
        lbl_dir = OUTPUT_DIR / "labels" / split
        if not img_dir.exists():
            continue
        n_img = len(list(img_dir.glob("*")))
        n_lbl = len(list(lbl_dir.glob("*.txt")))

        # 클래스별 카운트
        class_counts = {v[1]: 0 for v in CLASS_MAP.values()}
        for lbl in lbl_dir.glob("*.txt"):
            with open(lbl) as f:
                for line in f:
                    cid = int(line.strip().split()[0])
                    for name, (idx, label) in CLASS_MAP.items():
                        if idx == cid:
                            class_counts[label] += 1

        print(f"\n  [{split}]  이미지: {n_img}  라벨: {n_lbl}")
        for label, cnt in class_counts.items():
            print(f"    {label:<10}: {cnt}개 bbox")


# ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    datasets = download_open_images()
    convert_to_yolo(datasets)
    create_yaml()
    print_stats()

    print("\n완료. 다음 단계: python train_yolo.py")
