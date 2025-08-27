"""
Jupyter 노트북에서 사용할 수 있는 SA3D YOLO 자동화 함수들
"""

import subprocess
import time
import os
from IPython.display import display, HTML
import threading
import queue

def run_command_async(command, description="", timeout=None):
    """
    명령어를 비동기로 실행하는 함수
    Jupyter 노트북에서 사용하기 적합합니다.
    """
    print(f"\n{'='*60}")
    print(f"실행 중: {description}")
    print(f"명령어: {command}")
    print(f"{'='*60}")
    
    try:
        # subprocess.Popen을 사용하여 비동기 실행
        process = subprocess.Popen(
            command,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            universal_newlines=True
        )
        
        # 출력을 실시간으로 읽기
        output_lines = []
        while True:
            output = process.stdout.readline()
            if output == '' and process.poll() is not None:
                break
            if output:
                print(output.strip())
                output_lines.append(output.strip())
        
        # 프로세스 완료 대기
        return_code = process.wait()
        
        if return_code == 0:
            print(f"\n✅ {description} 완료!")
            return True
        else:
            print(f"\n❌ {description} 실패 (종료 코드: {return_code})")
            return False
            
    except subprocess.TimeoutExpired:
        print(f"\n⏰ {description} 시간 초과")
        process.kill()
        return False
    except Exception as e:
        print(f"\n❌ {description} 오류 발생: {e}")
        return False

def run_single_set_notebook(set_num):
    """
    Jupyter 노트북에서 특정 세트 하나를 실행하는 함수
    """
    print(f"🚀 Set{set_num} 실행을 시작합니다...")
    
    # 1단계: 기본 학습
    print(f"\n📚 Set{set_num} 기본 학습 시작")
    command1 = f"python run.py --config=configs/nerf_unbounded/Set{set_num}.py --stop_at=20000 --render_video --i_weights=10000"
    if not run_command_async(command1, f"Set{set_num} 기본 학습"):
        print("기본 학습 실패. 프로그램을 종료합니다.")
        return False
    
    print("⏳ 5초 대기 후 세그멘테이션 학습 시작...")
    time.sleep(5)
    
    # 2단계: 세그멘테이션 학습
    print(f"\n🎯 Set{set_num} 세그멘테이션 학습 시작")
    command2 = f"python run_seg_gui.py --config=configs/nerf_unbounded/seg_Set{set_num}.py --segment --sp_name=_gui --num_prompts=20 --render_opt=train --save_ckpt"
    if not run_command_async(command2, f"Set{set_num} 세그멘테이션 학습"):
        print("세그멘테이션 학습 실패. 프로그램을 종료합니다.")
        return False
    
    print("⏳ 5초 대기 후 렌더링 시작...")
    time.sleep(5)
    
    # 3단계: 렌더링 및 이미지 생성
    print(f"\n🎨 Set{set_num} 렌더링 및 이미지 생성 시작")
    command3 = f"python run_seg_gui.py --config=configs/nerf_unbounded/seg_Set{set_num}.py --segment --sp_name=_gui --num_prompts=20 --render_only --render_opt=video --dump_images --seg_type seg_img seg_density"
    if not run_command_async(command3, f"Set{set_num} 렌더링 및 이미지 생성"):
        print("렌더링 실패. 프로그램을 종료합니다.")
        return False
    
    print(f"\n🎉 Set{set_num} 모든 작업이 완료되었습니다!")
    return True

def run_batch_notebook(start_set=1, end_set=13):
    """
    Jupyter 노트북에서 배치 학습을 실행하는 함수
    """
    print(f"🚀 SA3D YOLO 자동화 배치 학습을 시작합니다...")
    print(f"📊 실행 범위: Set{start_set} ~ Set{end_set}")
    
    sets = list(range(start_set, end_set + 1))
    
    # 1단계: 기본 학습
    print(f"\n📚 1단계: 기본 학습 시작")
    for set_num in sets:
        command = f"python run.py --config=configs/nerf_unbounded/Set{set_num}.py --stop_at=20000 --render_video --i_weights=10000"
        description = f"Set{set_num} 기본 학습"
        
        if not run_command_async(command, description):
            print(f"⚠️ Set{set_num}에서 오류 발생. 다음 세트로 진행합니다.")
            continue
        
        print(f"⏳ Set{set_num} 완료 후 5초 대기...")
        time.sleep(5)
    
    # 2단계: 세그멘테이션 학습
    print(f"\n🎯 2단계: 세그멘테이션 학습 시작")
    for set_num in sets:
        command = f"python run_seg_gui.py --config=configs/nerf_unbounded/seg_Set{set_num}.py --segment --sp_name=_gui --num_prompts=20 --render_opt=train --save_ckpt"
        description = f"Set{set_num} 세그멘테이션 학습"
        
        if not run_command_async(command, description):
            print(f"⚠️ Set{set_num}에서 오류 발생. 다음 세트로 진행합니다.")
            continue
        
        print(f"⏳ Set{set_num} 완료 후 5초 대기...")
        time.sleep(5)
    
    # 3단계: 렌더링 및 이미지 생성
    print(f"\n🎨 3단계: 렌더링 및 이미지 생성 시작")
    for set_num in sets:
        command = f"python run_seg_gui.py --config=configs/nerf_unbounded/seg_Set{set_num}.py --segment --sp_name=_gui --num_prompts=20 --render_only --render_opt=video --dump_images --seg_type seg_img seg_density"
        description = f"Set{set_num} 렌더링 및 이미지 생성"
        
        if not run_command_async(command, description):
            print(f"⚠️ Set{set_num}에서 오류 발생. 다음 세트로 진행합니다.")
            continue
        
        print(f"⏳ Set{set_num} 완료 후 5초 대기...")
        time.sleep(5)
    
    print("\n🎉 모든 배치 학습이 완료되었습니다!")

def run_specific_sets_notebook(set_numbers):
    """
    Jupyter 노트북에서 특정 세트들만 실행하는 함수
    예시: run_specific_sets_notebook([1, 3, 5, 7])
    """
    print(f"🚀 선택된 세트들 실행: {set_numbers}")
    
    for set_num in set_numbers:
        if not run_single_set_notebook(set_num):
            print(f"⚠️ Set{set_num} 실행 실패. 다음 세트로 진행합니다.")
            continue
        
        if set_num != set_numbers[-1]:  # 마지막 세트가 아니면 대기
            print("⏳ 다음 세트 실행 전 10초 대기...")
            time.sleep(10)
    
    print("\n🎉 선택된 모든 세트 실행이 완료되었습니다!")

# 사용 예시를 위한 헬프 함수
def show_usage():
    """
    사용법을 보여주는 함수
    """
    usage_html = """
    <div style="background-color: #f0f0f0; padding: 15px; border-radius: 5px; margin: 10px 0;">
        <h3>🚀 SA3D YOLO 자동화 함수 사용법</h3>
        <p><strong>1. 특정 세트 하나 실행:</strong></p>
        <code>run_single_set_notebook(6)</code>
        
        <p><strong>2. 연속된 세트들 배치 실행:</strong></p>
        <code>run_batch_notebook(1, 13)  # Set1부터 Set13까지</code>
        <code>run_batch_notebook(6, 8)   # Set6부터 Set8까지</code>
        
        <p><strong>3. 특정 세트들만 실행:</strong></p>
        <code>run_specific_sets_notebook([1, 3, 5, 7])</code>
        
        <p><strong>4. 사용법 보기:</strong></p>
        <code>show_usage()</code>
    </div>
    """
    display(HTML(usage_html))

# 초기 사용법 표시
print("🚀 SA3D YOLO 자동화 함수들이 로드되었습니다!")
print("사용법을 보려면 'show_usage()'를 실행하세요.")
