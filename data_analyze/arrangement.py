#2번 코드

import os, re
import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm

# —————————— 사용자 설정 부분 ——————————
gt_folder = 'GT_rename_frames/set10_p6'
pred_folder = 'frames_only_YOLO/set10_p6'
min_iou_threshold = 0.01  # 최소 허용 IoU (0.0~1.0) → 필요에 따라 조정
# ————————————————————————————————
pred_pattern = re.compile(r'^(\d+)\.png$')

# 파일 리스트 로딩 (숫자 이름 기준 정렬)
gt_files = sorted(
    [f for f in os.listdir(gt_folder) if f.endswith('.png') and f[:-4].isdigit()],
    key=lambda x: int(x[:-4])
)
pred_files = sorted(
    [f for f in os.listdir(pred_folder) if pred_pattern.match(f)],
    key=lambda x: int(pred_pattern.match(x).group(1))
)

# 1단계: 모든 예측 이미지 이름 앞에 0000_ 접두사 추가 (충돌 방지)
print("🔄 1단계: 모든 예측 이미지에 0000_ 접두사 추가 중...")
temp_renames = []
for pred_fn in pred_files:
    src = os.path.join(pred_folder, pred_fn)
    temp_name = f"0000_{pred_fn}"
    temp_dst = os.path.join(pred_folder, temp_name)
    
    try:
        os.rename(src, temp_dst)
        temp_renames.append((temp_dst, pred_fn))
        print(f"  → {pred_fn} → {temp_name}")
    except Exception as e:
        print(f"  ❌ 임시 이름 변경 실패: {pred_fn} → {temp_name} (오류: {e})")

# 0000_ 접두사가 추가된 파일 리스트 다시 로딩
pred_files = sorted(
    [f for f in os.listdir(pred_folder) if f.startswith('0000_') and pred_pattern.match(f[5:])],
    key=lambda x: int(pred_pattern.match(x[5:]).group(1))
)

M = len(pred_files)
N = len(gt_files)

# IoU 계산 함수
def calc_iou(mask1, mask2):
    inter = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    return inter / union if union > 0 else 0.0

# IoU 행렬 생성 (M x N)
iou_matrix = np.zeros((M, N), dtype=float)
for i, pred_fn in enumerate(tqdm(pred_files, desc="IoU 계산")):
    pred = cv2.imread(os.path.join(pred_folder, pred_fn), cv2.IMREAD_GRAYSCALE)
    _, pred_bin = cv2.threshold(pred, 127, 1, cv2.THRESH_BINARY)
    for j, gt_fn in enumerate(gt_files):
        gt = cv2.imread(os.path.join(gt_folder, gt_fn), cv2.IMREAD_GRAYSCALE)
        # 크기 불일치 시 GT 리사이즈
        if gt.shape != pred.shape:
            gt = cv2.resize(gt, (pred.shape[1], pred.shape[0]), interpolation=cv2.INTER_NEAREST)
        _, gt_bin = cv2.threshold(gt, 127, 1, cv2.THRESH_BINARY)
        iou_matrix[i, j] = calc_iou(pred_bin, gt_bin)

# Hungarian 알고리즘으로 최적 매칭 (maximize IoU → minimize -IoU)
row_ind, col_ind = linear_sum_assignment(-iou_matrix)

# 임계치 없이 모든 결과 사용 (전역 1:1 매칭)
matches = list(zip(row_ind, col_ind))

# 매칭 결과 로깅
print("🧾 매칭 결과 (pred → gt, IoU):")
for pred_idx, gt_idx in matches:
    print(f"  {pred_files[pred_idx]} → {gt_files[gt_idx]} (IoU={iou_matrix[pred_idx, gt_idx]:.4f})")

# 2단계: 예측 마스크 파일명 → GT 파일명으로 리네임
print("🔄 2단계: 최종 GT 이름으로 변경 중...")
for pred_idx, gt_idx in matches:
    src = os.path.join(pred_folder, pred_files[pred_idx])
    dst = os.path.join(pred_folder, gt_files[gt_idx])
    try:
        os.rename(src, dst)
        print(f"  → {pred_files[pred_idx]} → {gt_files[gt_idx]}")
    except Exception as e:
        print(f"  ❌ 이름 변경 실패: {pred_files[pred_idx]} → {gt_files[gt_idx]} (오류: {e})")

# 결과 요약 출력
print(f"✅ 헝가리안 전역 최적 매칭으로 총 {len(matches)}개 쌍을 이름 변경했습니다.")
if M != N:
    if M < N:
        unused_gt = sorted(list(set(gt_files[i] for i in range(N)) - {gt_files[g] for _, g in matches}))
        print(f"→ 매칭되지 않은 GT ({len(unused_gt)}개): {unused_gt}")
    else:
        unused_pred = sorted(list(set(pred_files[i] for i in range(M)) - {pred_files[p] for p, _ in matches}))
        print(f"→ 매칭되지 않은 예측 마스크 ({len(unused_pred)}개): {unused_pred}")
