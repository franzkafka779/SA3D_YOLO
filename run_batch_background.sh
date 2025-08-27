#!/bin/bash

# SA3D YOLO 백그라운드 배치 학습 스크립트
# 사용법: ./run_batch_background.sh [start_set] [end_set]
# 예시: ./run_batch_background.sh 1 13 (모든 세트)
# 예시: ./run_batch_background.sh 6 8 (Set6부터 Set8까지)

# 로그 파일 설정
LOG_DIR="logs"
mkdir -p $LOG_DIR
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="$LOG_DIR/batch_${TIMESTAMP}.log"

# 시작/끝 세트 설정
START_SET=${1:-1}
END_SET=${2:-13}

echo "🚀 SA3D YOLO 백그라운드 배치 학습을 시작합니다..."
echo "📅 시작 시간: $(date)"
echo "📊 실행 범위: Set${START_SET} ~ Set${END_SET}"
echo "📝 로그 파일: $LOG_FILE"
echo ""

# 로그 함수
log_message() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

# 명령어 실행 함수
run_command() {
    local command="$1"
    local description="$2"
    local set_num="$3"
    
    log_message "=== Set${set_num}: ${description} 시작 ==="
    log_message "명령어: $command"
    
    # 명령어 실행 및 로그 저장
    if eval "$command" >> "$LOG_FILE" 2>&1; then
        log_message "✅ Set${set_num}: ${description} 완료"
        return 0
    else
        log_message "❌ Set${set_num}: ${description} 실패"
        return 1
    fi
}

# 메인 실행 루프
for ((set_num=START_SET; set_num<=END_SET; set_num++)); do
    log_message "🚀 Set${set_num} 실행 시작"
    
    # 1단계: 기본 학습
    command1="python run.py --config=configs/nerf_unbounded/Set${set_num}.py --stop_at=20000 --render_video --i_weights=10000"
    if ! run_command "$command1" "기본 학습" "$set_num"; then
        log_message "⚠️ Set${set_num} 기본 학습 실패, 다음 세트로 진행"
        continue
    fi
    
    # 5초 대기
    log_message "⏳ Set${set_num} 완료 후 5초 대기..."
    sleep 5
    
    # 2단계: 세그멘테이션 학습
    command2="python run_seg_gui.py --config=configs/nerf_unbounded/seg_Set${set_num}.py --segment --sp_name=_gui --num_prompts=20 --render_opt=train --save_ckpt"
    if ! run_command "$command2" "세그멘테이션 학습" "$set_num"; then
        log_message "⚠️ Set${set_num} 세그멘테이션 학습 실패, 다음 세트로 진행"
        continue
    fi
    
    # 5초 대기
    log_message "⏳ Set${set_num} 완료 후 5초 대기..."
    sleep 5
    
    # 3단계: 렌더링 및 이미지 생성
    command3="python run_seg_gui.py --config=configs/nerf_unbounded/seg_Set${set_num}.py --segment --sp_name=_gui --num_prompts=20 --render_only --render_opt=video --dump_images --seg_type seg_img seg_density"
    if ! run_command "$command3" "렌더링 및 이미지 생성" "$set_num"; then
        log_message "⚠️ Set${set_num} 렌더링 실패, 다음 세트로 진행"
        continue
    fi
    
    log_message "🎉 Set${set_num} 모든 작업 완료!"
    log_message "----------------------------------------"
done

log_message "🎉 모든 배치 학습이 완료되었습니다!"
log_message "📅 종료 시간: $(date)"
echo ""
echo "🎉 배치 학습이 완료되었습니다!"
echo "📝 로그 파일: $LOG_FILE"
echo "📊 실행된 세트: Set${START_SET} ~ Set${END_SET}"
