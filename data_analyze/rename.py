import os
import shutil

# [사용자 설정]
input_folder = 'GT_frames/set1_p2'     # 기존 마스크 폴더
output_folder = 'GT_rename_frames/set1_p2'     # 새로 저장할 폴더
person_id = 'p2'                    # 현재 파일에 포함된 사람 ID

# 위에서 주신 순서
original_order = [
    "01", "02", "03", "04", "05", "06", "07", "08",
    "21", "22", "23", "24", "25", "26", "27", "28",
    "29", "30", "31", "32", "33", "34", "35", "36",
    "09", "10", "11", "12", "13", "14", "15", "16"
]

# 출력 폴더 생성
os.makedirs(output_folder, exist_ok=True)

# 순서대로 파일명 바꾸기
for new_index, id_num in enumerate(original_order):
    original_name = f"{id_num}_{person_id}.png"
    original_path = os.path.join(input_folder, original_name)

    if not os.path.exists(original_path):
        print(f"[⚠️ 누락됨] 파일 없음: {original_name}")
        continue

    new_name = f"{new_index}.png"
    new_path = os.path.join(output_folder, new_name)

    shutil.copyfile(original_path, new_path)
    print(f"[변경됨] {original_name} → {new_name}")

print("✅ 파일 이름 재정렬 완료.")