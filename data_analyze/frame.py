#1번 코드

import cv2
import os

# [사용자 설정]
video_path = 'SA3D_only_YOLO/truck/render_train_coarse_segmentation_gui/video.segseg_gui_seg_img.mp4'     # 동영상 파일 경로
output_dir = 'frames_only_YOLO/truck'       # 저장할 폴더 경로

os.makedirs(output_dir, exist_ok=True)

# 비디오 캡처 시작
cap = cv2.VideoCapture(video_path)
frame_index = 0

while True:
    success, frame = cap.read()
    if not success:
        break

    # 저장 경로 및 이름 (번호만)
    filename = os.path.join(output_dir, f"{frame_index}.png")
    cv2.imwrite(filename, frame)
    print(f"[저장됨] {filename}")

    frame_index += 1

cap.release()
print(f"✅ 총 {frame_index}개의 프레임이 저장되었습니다.")