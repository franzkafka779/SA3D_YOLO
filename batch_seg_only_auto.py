#!/usr/bin/env python3
"""
2번째 셀만 자동화하는 스크립트
세그멘테이션 학습 결과를 사용해서 렌더링만 자동으로 수행
instance 선택은 기존 GUI 방식을 유지하되 자동화
"""

import subprocess
import time
import os
import sys
from pathlib import Path

def run_command(command, description=""):
    """명령어를 실행하고 결과를 반환합니다."""
    print(f"\n{'='*60}")
    print(f"실행 중: {description}")
    print(f"명령어: {command}")
    print(f"{'='*60}")
    
    try:
        # subprocess.run을 사용하여 명령어 실행
        result = subprocess.run(
            command, 
            shell=True, 
            check=True,
            capture_output=False,  # 실시간 출력을 위해 False로 설정
            text=True
        )
        print(f"\n✅ {description} 완료!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n❌ {description} 실패: {e}")
        return False
    except KeyboardInterrupt:
        print(f"\n⚠️ {description} 사용자에 의해 중단됨")
        return False

def main():
    """메인 실행 함수"""
    print("🚀 SA3D YOLO 2번째 셀 자동화를 시작합니다...")
    print("💡 세그멘테이션 학습 결과를 사용해서 렌더링만 자동으로 수행합니다.")
    print("⚠️  주의: 이 스크립트를 실행하기 전에 1번째 셀(세그멘테이션 학습)이 완료되어야 합니다!")
    
    # 사용자 확인
    response = input("\n1번째 셀(세그멘테이션 학습)이 완료되었습니까? (y/n): ")
    if response.lower() != 'y':
        print("1번째 셀을 먼저 완료한 후 다시 실행해주세요.")
        return
    
    # Set1부터 Set13까지 순차 실행 (2번째 셀만)
    sets = list(range(1, 14))
    
    print(f"\n🎨 2번째 셀: 렌더링 및 이미지 생성 시작")
    print("📊 실행 범위: Set1 ~ Set13")
    
    for set_num in sets:
        print(f"\n🚀 Set{set_num} 렌더링 시작...")
        
        # 2번째 셀 명령어 (렌더링만) - 자동화된 스크립트 사용
        command = f"python run_seg_gui_auto_interaction.py --config=configs/nerf_unbounded/seg_Set{set_num}.py --segment --sp_name=_gui --num_prompts=20 --render_opt=train --save_ckpt"
        description = f"Set{set_num} 렌더링 및 이미지 생성 (자동화)"
        
        if not run_command(command, description):
            print(f"⚠️ Set{set_num}에서 오류 발생. 다음 세트로 진행합니다.")
            continue
        
        print(f"⏳ Set{set_num} 완료 후 5초 대기...")
        time.sleep(5)
    
    print("\n🎉 모든 세트의 렌더링이 완료되었습니다!")
    print("💡 2번째 셀의 모든 작업이 자동으로 수행되었습니다.")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️ 사용자에 의해 프로그램이 중단되었습니다.")
        sys.exit(1)
