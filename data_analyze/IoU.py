#3번 코드

import os
import re
import cv2
import numpy as np
import pandas as pd

# ————— 사용자 설정 —————
gt_root      = 'Truck/output/GT'             # GT 마스크 폴더
pred_root    = 'frames_only_YOLO/truck'  # 예측 마스크 폴더
output_excel = 'onlyYOLO_truck.xlsx'         # 출력 엑셀 파일
# ————————————————————

# 6자리 숫자 파일만 매칭하는 정규식
mask_pattern = re.compile(r'^(\d+)\.(png|jpg|jpeg)$')

# GT 파일 목록 (숫자 순서대로 정렬)
gt_files = sorted(
    [f for f in os.listdir(gt_root) if mask_pattern.match(f)],
    key=lambda x: int(mask_pattern.match(x).group(1))
)

results       = []  # 계산 결과 저장
missing_preds = []  # 예측이 없는 파일
size_errors   = []  # 리사이즈 후에도 크기 불일치 파일

for fn in gt_files:
    gt_path   = os.path.join(gt_root, fn)
    pred_path = os.path.join(pred_root, fn)

    # 1) 예측 마스크 누락 시
    if not os.path.exists(pred_path):
        missing_preds.append(fn)
        results.append({'filename': fn, 'IoU': '-'})
        continue

    # 2) 이미지 로드
    gt_mask   = cv2.imread(gt_path,   cv2.IMREAD_GRAYSCALE)
    pred_mask = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE)

    # 3) 크기 맞추기
    if gt_mask.shape != pred_mask.shape:
        gt_mask = cv2.resize(
            gt_mask,
            (pred_mask.shape[1], pred_mask.shape[0]),
            interpolation=cv2.INTER_NEAREST
        )

    # 4) 여전히 크기 불일치 시
    if gt_mask.shape != pred_mask.shape:
        size_errors.append(fn)
        results.append({'filename': fn, 'IoU': '-'})
        continue

    # 5) 이진화 & IoU 계산
    _, gt_bin   = cv2.threshold(gt_mask,   127, 1, cv2.THRESH_BINARY)
    _, pred_bin = cv2.threshold(pred_mask, 127, 1, cv2.THRESH_BINARY)
    inter = np.logical_and(gt_bin, pred_bin).sum()
    uni   = np.logical_or (gt_bin, pred_bin).sum()
    iou   = inter / uni if uni > 0 else 0.0

    results.append({'filename': fn, 'IoU': round(iou, 4)})

# DataFrame 생성 및 엑셀 저장
df = pd.DataFrame(results)
with pd.ExcelWriter(output_excel, engine='openpyxl') as writer:
    df.to_excel(writer, sheet_name='IoU', index=False)

# 요약 출력
print(f"✅ 처리 완료: 총 {len(results)}개 행 생성")
if missing_preds:
    print(f"  ⚠️ 예측 누락: {len(missing_preds)}개 → {missing_preds}")
if size_errors:
    print(f"  ⚠️ 크기 불일치: {len(size_errors)}개 → {size_errors}")
print(f"결과 파일: {output_excel}")
