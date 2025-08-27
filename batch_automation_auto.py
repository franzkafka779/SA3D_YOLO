#!/usr/bin/env python3
"""
완전 자동화된 SA3D YOLO 배치 학습 스크립트
GUI 상호작용 없이 모든 과정을 자동으로 수행합니다.
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
    print("🚀 SA3D YOLO 완전 자동화 배치 학습을 시작합니다...")
    print("💡 GUI 상호작용 없이 모든 과정을 자동으로 수행합니다.")
    
    # Set1부터 Set13까지 순차 실행
    sets = list(range(1, 14))
    
    # 1단계: 기본 학습 (run.py)
    print("\n📚 1단계: 기본 학습 시작")
    for set_num in sets:
        command = f"python run.py --config=configs/nerf_unbounded/Set{set_num}.py --stop_at=20000 --render_video --i_weights=10000"
        description = f"Set{set_num} 기본 학습"
        
        if not run_command(command, description):
            print(f"⚠️ Set{set_num}에서 오류 발생. 다음 세트로 진행합니다.")
            continue
        
        print(f"⏳ Set{set_num} 완료 후 5초 대기...")
        time.sleep(5)
    
    # 2단계: 자동 세그멘테이션 학습 (run_seg_auto.py 사용)
    print("\n🎯 2단계: 자동 세그멘테이션 학습 시작")
    for set_num in sets:
        command = f"python run_seg_auto.py --config=configs/nerf_unbounded/seg_Set{set_num}.py --segment --sp_name=_gui --num_prompts=20 --render_opt=train --save_ckpt"
        description = f"Set{set_num} 자동 세그멘테이션 학습"
        
        if not run_command(command, description):
            print(f"⚠️ Set{set_num}에서 오류 발생. 다음 세트로 진행합니다.")
            continue
        
        print(f"⏳ Set{set_num} 완료 후 5초 대기...")
        time.sleep(5)
    
    # 3단계: 렌더링 및 이미지 생성 (자동 세그멘테이션 스크립트 사용)
    print("\n🎨 3단계: 렌더링 및 이미지 생성 시작")
    for set_num in sets:
        command = f"python run_seg_auto.py --config=configs/nerf_unbounded/seg_Set{set_num}.py --segment --sp_name=_gui --num_prompts=20 --render_only --render_opt=video --dump_images --seg_type seg_img seg_density"
        description = f"Set{set_num} 렌더링 및 이미지 생성"
        
        if not run_command(command, description):
            print(f"⚠️ Set{set_num}에서 오류 발생. 다음 세트로 진행합니다.")
            continue
        
        print(f"⏳ Set{set_num} 완료 후 5초 대기...")
        time.sleep(5)
    
    print("\n🎉 모든 배치 학습이 완료되었습니다!")
    print("💡 GUI 상호작용 없이 모든 과정이 자동으로 수행되었습니다.")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️ 사용자에 의해 프로그램이 중단되었습니다.")
        sys.exit(1)
