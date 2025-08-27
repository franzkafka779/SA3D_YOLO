#!/usr/bin/env python3
"""
특정 세트 하나만 실행하는 스크립트
사용법: python run_single_set.py <set_number>
예시: python run_single_set.py 6
"""

import subprocess
import sys
import time

def run_command(command, description=""):
    """명령어를 실행하고 결과를 반환합니다."""
    print(f"\n{'='*60}")
    print(f"실행 중: {description}")
    print(f"명령어: {command}")
    print(f"{'='*60}")
    
    try:
        result = subprocess.run(
            command, 
            shell=True, 
            check=True,
            capture_output=False,
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
    if len(sys.argv) != 2:
        print("사용법: python run_single_set.py <set_number>")
        print("예시: python run_single_set.py 6")
        sys.exit(1)
    
    try:
        set_num = int(sys.argv[1])
        if set_num < 1 or set_num > 13:
            print("세트 번호는 1-13 사이여야 합니다.")
            sys.exit(1)
    except ValueError:
        print("올바른 숫자를 입력해주세요.")
        sys.exit(1)
    
    print(f"🚀 Set{set_num} 실행을 시작합니다...")
    
    # 1단계: 기본 학습
    print(f"\n📚 Set{set_num} 기본 학습 시작")
    command1 = f"python run.py --config=configs/nerf_unbounded/Set{set_num}.py --stop_at=20000 --render_video --i_weights=10000"
    if not run_command(command1, f"Set{set_num} 기본 학습"):
        print("기본 학습 실패. 프로그램을 종료합니다.")
        sys.exit(1)
    
    print("⏳ 5초 대기 후 세그멘테이션 학습 시작...")
    time.sleep(5)
    
    # 2단계: 세그멘테이션 학습
    print(f"\n🎯 Set{set_num} 세그멘테이션 학습 시작")
    command2 = f"python run_seg_gui.py --config=configs/nerf_unbounded/seg_Set{set_num}.py --segment --sp_name=_gui --num_prompts=20 --render_opt=train --save_ckpt"
    if not run_command(command2, f"Set{set_num} 세그멘테이션 학습"):
        print("세그멘테이션 학습 실패. 프로그램을 종료합니다.")
        sys.exit(1)
    
    print("⏳ 5초 대기 후 렌더링 시작...")
    time.sleep(5)
    
    # 3단계: 렌더링 및 이미지 생성
    print(f"\n🎨 Set{set_num} 렌더링 및 이미지 생성 시작")
    command3 = f"python run_seg_gui.py --config=configs/nerf_unbounded/seg_Set{set_num}.py --segment --sp_name=_gui --num_prompts=20 --render_only --render_opt=video --dump_images --seg_type seg_img seg_density"
    if not run_command(command3, f"Set{set_num} 렌더링 및 이미지 생성"):
        print("렌더링 실패. 프로그램을 종료합니다.")
        sys.exit(1)
    
    print(f"\n🎉 Set{set_num} 모든 작업이 완료되었습니다!")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️ 사용자에 의해 프로그램이 중단되었습니다.")
        sys.exit(1)
