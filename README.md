# 🎯 비대면 면접 부정행위 탐지 시스템

> AI 기반 비대면 면접 영상 분석으로 부정행위를 자동 감지하는 시스템

---

## 📌 개요

비대면 면접에서 발생할 수 있는 다양한 부정행위를 AI로 탐지합니다.  
시선 추적, 반복 행동 패턴 분석, 객체 감지, 음성 분석을 통합한 파이프라인으로 동작합니다.

---

## 🔍 탐지 항목

| 유형 | 방법 |
|------|------|
| 👀 시선 이탈 | Head Pose Estimation (yaw/pitch) + Eye Tracking |
| 🔁 반복 행동 | 연속 응시 편향 패턴 분석 |
| 📱 금지 물체 | YOLOv11 커스텀 모델 (책, 핸드폰, 태블릿, 노트북) |
| 🎤 외부 목소리 | resemblyzer 화자 분리 + 코사인 유사도 |

---

## 🗂️ 파일 구조

```
├── final_pipeline.py        # 전체 파이프라인 통합 실행
├── extract_features.py      # 영상에서 head pose / eye 피처 추출
├── event_detection.py       # 시선 이탈 이벤트 탐지
├── repetition_analysis.py   # 반복 행동 패턴 분석
├── audio_analysis.py        # 외부 목소리 감지
├── train_yolo.py            # YOLOv11 커스텀 파인튜닝
├── compute_normal_ranges.py # 정상 범위 기준값 산출
├── evaluate_results.py      # 탐지 결과 평가
├── visualize_tracking.py    # 추적 결과 시각화
├── render_warning_video.py  # 경고 영상 렌더링
├── render_analysis_video.py # 분석 결과 영상 렌더링
├── draw_pipeline.py         # 파이프라인 구조 시각화
└── run_test_videos.py       # 테스트 영상 일괄 실행
```

---

## ⚙️ 실행 방법

### 1. 의존성 설치

```bash
pip install opencv-python mediapipe ultralytics resemblyzer pandas numpy
```

### 2. 정상 범위 기준값 산출 (최초 1회)

```bash
python compute_normal_ranges.py
```

### 3. 부정행위 탐지 실행

```bash
python final_pipeline.py --video path/to/video.mp4
```

### 4. 결과 확인

```
data/results/<video_id>_final_report.csv  # 시간대별 이상 이벤트
data/results/<video_id>_summary.txt       # 요약 리포트
```

---

## 🛠️ 주요 기술 스택

- **YOLOv11** — 실시간 객체 탐지
- **MediaPipe** — Head Pose Estimation, Eye Tracking
- **resemblyzer** — 화자 임베딩 기반 음성 분리
- **ByteTrack** — 객체 추적
- **OpenCV** — 영상 처리

---

## 📊 파이프라인 구조

```
입력 영상
   ├── [시각 분석] Head Pose + Eye Tracking → 시선 이탈 이벤트
   ├── [객체 탐지] YOLOv11 → 금지 물체 감지
   └── [음성 분석] resemblyzer → 외부 목소리 감지
            ↓
    통합 리포트 생성 (CSV + TXT)
```
