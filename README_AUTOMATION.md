# SA3D YOLO 자동화 스크립트 사용법

이 문서는 SA3D YOLO의 배치 학습을 자동화하기 위한 다양한 스크립트들의 사용법을 설명합니다.

## 🚀 문제 상황

기존 `batch.ipynb` 파일에서는 각 세트별 학습을 수행할 때 자동 종료가 되지 않아 수동으로 인터럽트를 걸어야 하는 문제가 있었습니다.

## 💡 해결 방안

### 1. Python 스크립트 자동화 (`batch_automation.py`)

**특징:**
- 모든 세트(Set1~Set13)를 순차적으로 자동 실행
- 각 단계별로 오류 처리 및 로깅
- 실시간 진행 상황 모니터링

**사용법:**
```bash
# 모든 세트 실행
python batch_automation.py

# 백그라운드 실행 (로그 파일에 출력 저장)
nohup python batch_automation.py > batch.log 2>&1 &
```

### 2. 개별 세트 실행 (`run_single_set.py`)

**특징:**
- 특정 세트 하나만 실행
- 명령행 인자로 세트 번호 지정
- 오류 발생 시 즉시 종료

**사용법:**
```bash
# Set6 실행
python run_single_set.py 6

# Set1 실행
python run_single_set.py 1
```

### 3. 백그라운드 배치 실행 (`run_batch_background.sh`)

**특징:**
- 쉘 스크립트로 백그라운드 실행
- 자동 로그 파일 생성
- 범위 지정 가능 (시작 세트 ~ 끝 세트)

**사용법:**
```bash
# 실행 권한 부여
chmod +x run_batch_background.sh

# 모든 세트 실행 (Set1~Set13)
./run_batch_background.sh

# 특정 범위 실행 (Set6~Set8)
./run_batch_background.sh 6 8

# 백그라운드 실행
nohup ./run_batch_background.sh > background.log 2>&1 &
```

### 4. Jupyter 노트북 자동화 (`batch_automation_notebook.py`)

**특징:**
- Jupyter 노트북에서 직접 사용 가능
- 비동기 실행으로 노트북 블로킹 방지
- 다양한 실행 옵션 제공

**사용법:**
```python
# 노트북에서 모듈 로드
%run batch_automation_notebook.py

# 특정 세트 하나 실행
run_single_set_notebook(6)

# 연속된 세트들 배치 실행
run_batch_notebook(1, 13)  # Set1부터 Set13까지
run_batch_notebook(6, 8)   # Set6부터 Set8까지

# 특정 세트들만 실행
run_specific_sets_notebook([1, 3, 5, 7])

# 사용법 보기
show_usage()
```

## 📋 실행 단계

모든 스크립트는 다음 3단계를 순차적으로 실행합니다:

1. **기본 학습** (`run.py`)
   - NeRF 모델 학습
   - 비디오 렌더링
   - 가중치 저장

2. **세그멘테이션 학습** (`run_seg_gui.py`)
   - 세그멘테이션 모델 학습
   - 체크포인트 저장

3. **렌더링 및 이미지 생성** (`run_seg_gui.py`)
   - 세그멘테이션 결과 렌더링
   - 이미지 파일 생성

## 🔧 설정 옵션

**공통 매개변수:**
- `--stop_at=20000`: 20,000 스텝에서 학습 중단
- `--render_video`: 비디오 렌더링 활성화
- `--i_weights=10000`: 10,000 스텝마다 가중치 저장
- `--num_prompts=20`: 20개의 프롬프트 사용
- `--sp_name=_gui`: 세션 이름 설정

## 📊 모니터링 및 로깅

### 로그 파일 위치
- **Python 스크립트**: 콘솔 출력
- **쉘 스크립트**: `logs/batch_YYYYMMDD_HHMMSS.log`
- **백그라운드 실행**: `nohup.out` 또는 지정된 로그 파일

### 진행 상황 확인
```bash
# 백그라운드 프로세스 확인
ps aux | grep python

# 로그 파일 실시간 모니터링
tail -f logs/batch_*.log

# 특정 세트 진행 상황 확인
grep "Set6" logs/batch_*.log
```

## ⚠️ 주의사항

1. **메모리 관리**: 각 세트 실행 후 5초 대기로 메모리 정리
2. **오류 처리**: 한 세트에서 오류 발생 시 다음 세트로 자동 진행
3. **중단 처리**: `Ctrl+C`로 언제든지 중단 가능
4. **리소스 모니터링**: GPU 메모리 및 CPU 사용량 주의

## 🚨 문제 해결

### 일반적인 문제들

**1. 권한 오류**
```bash
chmod +x run_batch_background.sh
```

**2. Python 경로 문제**
```bash
# 가상환경 활성화 확인
conda activate sa3d_yolo
# 또는
source activate sa3d_yolo
```

**3. 메모리 부족**
- 세트 간 대기 시간 증가 (5초 → 10초)
- 동시 실행 세트 수 제한

**4. 로그 파일 크기 제한**
```bash
# 로그 로테이션 설정
logrotate -f /etc/logrotate.conf
```

## 📞 지원

문제가 발생하거나 추가 기능이 필요한 경우:
1. 로그 파일 확인
2. 오류 메시지 분석
3. 시스템 리소스 상태 점검
4. 필요시 스크립트 수정

---

**🎯 목표**: 수동 인터럽트 없이 모든 세트의 학습을 자동으로 완료하는 것
