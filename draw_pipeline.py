import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import matplotlib.font_manager as fm
import os

for f in ['/System/Library/Fonts/Supplemental/AppleGothic.ttf',
          '/Library/Fonts/AppleGothic.ttf',
          '/System/Library/Fonts/AppleSDGothicNeo.ttc']:
    if os.path.exists(f):
        plt.rcParams['font.family'] = fm.FontProperties(fname=f).get_name()
        break

fig, ax = plt.subplots(figsize=(16, 9))
fig.patch.set_facecolor('#EBF0F9')
ax.set_facecolor('#EBF0F9')
ax.set_xlim(0, 16)
ax.set_ylim(0, 9)
ax.axis('off')

def rbox(ax, x, y, w, h, fc, ec='none', lw=2.5, r=0.25, zorder=2):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
        boxstyle=f"round,pad=0,rounding_size={r}",
        facecolor=fc, edgecolor=ec, linewidth=lw, zorder=zorder))

def acc(ax, x, y, h, color):
    rbox(ax, x, y, 0.08, h, color, color, r=0.03, zorder=3)

def arrow(ax, x1, y1, x2, y2, color='#9AAAC8', lw=2.2):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
        arrowprops=dict(arrowstyle='->', color=color, lw=lw, mutation_scale=18), zorder=4)

def ctext(ax, cx, cy, txt, size, color='#111111'):
    ax.text(cx, cy, txt, ha='center', va='center',
            fontsize=size, fontweight='bold', color=color, zorder=3)

# ── 타이틀 ──
ax.text(0.45, 8.55, '제안 시스템 파이프라인', fontsize=22, fontweight='bold', color='#1B3A8A')
ax.text(0.45, 8.1, '입력 영상  →  3모듈 병렬 분석  →  통합 판정  →  경고 출력', fontsize=12, fontweight='bold', color='#6B7FA8')

# ── 좌측 강조선 / 하단 바 ──
rbox(ax, 0.3, 0.38, 0.07, 7.55, '#2D5BCA', '#2D5BCA', r=0.03, zorder=5)
rbox(ax, 0, 0,    16, 0.19, '#2D5BCA', zorder=5)
rbox(ax, 0, 0.19, 16, 0.12, '#7BAAF5', zorder=5)

# ══════════════════════════════
# ROW1: 입력영상 / 전처리
# ══════════════════════════════
rbox(ax, 5.1, 7.0, 5.8, 0.82, '#DBEAFE', '#93C5FD', lw=3)
acc(ax, 5.1, 7.0, 0.82, '#2D5BCA')
ctext(ax, 8.0, 7.44, '입력 영상', 16)
ctext(ax, 8.0, 7.13, '비대면 면접 영상 / 음성 스트림', 12, '#334155')

arrow(ax, 8.0, 7.0, 8.0, 6.52)

rbox(ax, 5.1, 5.65, 5.8, 0.78, '#EFF6FF', '#BFDBFE', lw=3)
acc(ax, 5.1, 5.65, 0.78, '#2D5BCA')
ctext(ax, 8.0, 6.07, '전처리', 16)
ctext(ax, 8.0, 5.80, '프레임 추출  |  음성 분리', 12, '#334155')

# ══════════════════════════════
# ROW2: 3개 모듈
# ══════════════════════════════
arrow(ax, 8.0, 5.65, 3.0,  4.92)
arrow(ax, 8.0, 5.65, 8.0,  4.92)
arrow(ax, 8.0, 5.65, 13.0, 4.92)

modules = [
    (0.5,  3.55, '#EFF6FF', '#93C5FD', '#3B82F6', '시각 분석 모듈',
     ['Head Pose', '시선 추적', '홍채 인식']),
    (5.35, 3.55, '#DCFCE7', '#6EE7B7', '#16A34A', '음성 분석 모듈',
     ['Resemblyzer', '코사인 유사도', '슬라이딩 윈도우']),
    (10.2, 3.55, '#FEF3C7', '#FCD34D', '#D97706', '객체 탐지 모듈',
     ['YOLOv11n', '핸드폰 / 노트북 / 책', 'Custom 모델']),
]

for mx, my, bg, border, ac, title, lines in modules:
    W = 4.9
    rbox(ax, mx, my, W, 1.25, 'white', border, lw=3)
    rbox(ax, mx, my+0.88, W, 0.37, bg, 'none', zorder=2)
    acc(ax, mx, my, 1.25, ac)
    cx = mx + W/2
    ctext(ax, cx, my+1.065, title, 14)
    for j, line in enumerate(lines):
        ctext(ax, cx, my+0.67 - j*0.27, line, 12, '#1e293b')

# ══════════════════════════════
# ROW3: 3 + 4 가운데 정렬
# ══════════════════════════════
BW = 5.0
GAP = 0.6
B3X = (16 - BW*2 - GAP) / 2
B4X = B3X + BW + GAP
B34Y = 2.05
MID_X = 8.0

arrow(ax, 3.0,  3.55, B3X+BW/2, B34Y+0.92)
arrow(ax, 8.0,  3.55, MID_X,    B34Y+0.92)
arrow(ax, 13.0, 3.55, B4X+BW/2, B34Y+0.92)

for bx, num, title, sub in [(B3X, 3, '이벤트 감지', '패턴 분석 / 이상 탐지'),
                              (B4X, 4, '통합 판정',   'OR 결합 로직')]:
    rbox(ax, bx, B34Y, BW, 0.92, '#FEF9C3', '#FCD34D', lw=3)
    ax.add_patch(plt.Circle((bx+0.38, B34Y+0.46), 0.23, color='#F59E0B', zorder=3))
    ctext(ax, bx+0.38, B34Y+0.46, str(num), 14, 'white')
    ctext(ax, bx+BW/2+0.2, B34Y+0.62, title, 14)
    ctext(ax, bx+BW/2+0.2, B34Y+0.27, sub, 12, '#92400E')

arrow(ax, B3X+BW, B34Y+0.46, B4X, B34Y+0.46, color='#F59E0B', lw=2.5)

# ══════════════════════════════
# 5: 부정행위 알림 가운데
# ══════════════════════════════
B5W = 4.6
B5X = (16 - B5W) / 2
B5Y = 0.72

arrow(ax, MID_X, B34Y, MID_X, B5Y+0.92, color='#EF4444', lw=2.5)

rbox(ax, B5X, B5Y, B5W, 0.92, '#FEE2E2', '#FCA5A5', lw=3)
ax.add_patch(plt.Circle((B5X+0.38, B5Y+0.46), 0.23, color='#EF4444', zorder=3))
ctext(ax, B5X+0.38, B5Y+0.46, '5', 14, 'white')
ctext(ax, B5X+B5W/2+0.2, B5Y+0.62, '부정행위 알림', 14)
ctext(ax, B5X+B5W/2+0.2, B5Y+0.27, '의심 감지 및 경고', 12, '#991B1B')

plt.tight_layout(pad=0)
plt.savefig('/Users/apple/Desktop/event_train/pipeline.png', dpi=180, bbox_inches='tight',
            facecolor='#EBF0F9')
print("저장 완료")
