# SA3D YOLO 완전 자동화 스크립트 사용법

이 문서는 SA3D YOLO의 배치 학습을 **GUI 상호작용 없이 완전 자동화**하기 위한 스크립트들의 사용법을 설명합니다.

## 🚀 핵심 문제 해결

### 기존 문제점
- `batch.ipynb`에서 각 세트별 학습 시 수동 인터럽트 필요
- `run_seg_gui.py`에서 GUI 창을 통한 사용자 상호작용 필요
- **자동화의 의미가 없었음**

### 해결 방안
- GUI 상호작용 부분을 완전 제거
- YOLO를 사용한 자동 마스크 생성
- 모든 과정을 명령행에서 자동 실행

## 💡 완전 자동화 스크립트

### 1. 핵심 자동화 스크립트 (`run_seg_auto.py`)

**특징:**
- GUI 없이 자동으로 세그멘테이션 수행
- YOLO를 사용하여 객체 자동 감지 및 마스크 생성
- 사용자 상호작용 완전 제거

**기술적 구현:**
```python
# YOLO로 자동 객체 감지
results = model_yolo.predict(source=image, imgsz=(h, w), classes=0)

# 자동 마스크 생성
masks = auto_generate_masks(init_rgb, num_prompts)

# GUI 없이 세그멘테이션 수행
auto_run_segmentation(Seg3d, num_prompts)
```

### 2. 완전 자동화 배치 실행 (`batch_automation_auto.py`)

**특징:**
- 모든 세트(Set1~Set13)를 GUI 없이 자동 실행
- `run_seg_auto.py` 사용으로 완전 자동화
- 수동 인터럽트 불필요

**사용법:**
```bash
# 모든 세트 완전 자동 실행
python batch_automation_auto.py

# 백그라운드 실행
nohup python batch_automation_auto.py > auto_batch.log 2>&1 &
```

### 3. 개별 세트 자동 실행 (`run_single_set_auto.py`)

**특징:**
- 특정 세트 하나를 GUI 없이 자동 실행
- 명령행 인자로 세트 번호 지정

**사용법:**
```bash
# Set6 자동 실행
python run_single_set_auto.py 6

# Set1 자동 실행
python run_single_set_auto.py 1
```

## 🔧 자동화 원리

### GUI 제거 과정
1. **기존 `run_seg_gui.py` 분석**
   - `Sam3dGUI.run()` 메서드에서 Dash 웹앱 실행
   - 사용자 클릭/입력 대기로 인한 블로킹

2. **자동화 구현**
   - `run_seg_auto.py`에서 GUI 클래스 제거
   - YOLO 기반 자동 마스크 생성
   - 배치 처리로 모든 프롬프트 자동 처리

### YOLO 자동 마스크 생성
```python
def auto_generate_masks(image, num_prompts=20):
    # YOLO로 객체 감지 (사람 클래스)
    results = model_yolo.predict(source=image, imgsz=(h, w), classes=0)
    
    # 마스크 품질 필터링
    if 0.01 < mask_area / total_area < 0.8:
        masks.append(mask)
    
    return masks[:num_prompts]
```

## 📋 실행 단계 (완전 자동화)

### 1단계: 기본 학습
- NeRF 모델 학습 (20,000 스텝)
- 비디오 렌더링
- 가중치 저장

### 2단계: 자동 세그멘테이션 학습
- **GUI 없음** - YOLO로 자동 마스크 생성
- 20개 프롬프트 자동 처리
- 체크포인트 자동 저장

### 3단계: 자동 렌더링
- 세그멘테이션 결과 자동 렌더링
- 이미지 파일 자동 생성
- **사용자 상호작용 불필요**

## 🚀 사용 예시

### 전체 배치 자동화
```bash
# 1. 완전 자동화 스크립트 실행
python batch_automation_auto.py

# 2. 백그라운드에서 실행 (권장)
nohup python batch_automation_auto.py > full_auto.log 2>&1 &

# 3. 진행 상황 모니터링
tail -f full_auto.log
```

### 특정 세트만 자동화
```bash
# Set6만 자동 실행
python run_single_set_auto.py 6

# Set1-5 범위만 실행 (스크립트 수정 필요)
for i in {1..5}; do
    python run_single_set_auto.py $i
done
```

## ⚠️ 주의사항 및 제한사항

### 현재 구현 상태
- **기본 자동화 완료**: GUI 상호작용 제거
- **마스크 생성 자동화**: YOLO 기반 객체 감지
- **일부 고급 기능**: 수동 조정이 필요한 세부 설정들

### 제한사항
1. **마스크 품질**: YOLO 감지 품질에 의존
2. **세부 조정**: 특정 객체에 대한 정밀한 마스크 조정 불가
3. **복잡한 장면**: 여러 객체가 겹친 복잡한 장면에서 정확도 저하 가능

### 개선 방향
1. **다중 클래스 지원**: 사람 외 다른 객체 클래스 추가
2. **마스크 품질 향상**: 더 정교한 마스크 생성 알고리즘
3. **적응형 프롬프트**: 장면 복잡도에 따른 자동 프롬프트 조정

## 🔍 문제 해결

### 일반적인 문제들

**1. YOLO 모델 로딩 실패**
```bash
# YOLO 모델 파일 확인
ls -la yolov8x-seg.pt

# 모델 다운로드 (필요시)
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8x-seg.pt
```

**2. 메모리 부족**
```bash
# GPU 메모리 확인
nvidia-smi

# 배치 크기 조정 (config 파일에서)
# cfg.coarse_train.batch_size = 4096  # 더 작은 값으로 조정
```

**3. 세그멘테이션 품질 저하**
- `num_prompts` 값을 조정 (20 → 10 또는 30)
- 마스크 품질 임계값 조정 (0.01-0.8 범위)

## 📊 모니터링 및 로깅

### 로그 파일 구조
```
auto_batch.log          # 전체 배치 실행 로그
auto_mask_01.png        # 자동 생성된 마스크 1
auto_mask_02.png        # 자동 생성된 마스크 2
...
```

### 진행 상황 확인
```bash
# 실시간 로그 모니터링
tail -f auto_batch.log

# 특정 세트 진행 상황
grep "Set6" auto_batch.log

# 오류 발생 여부 확인
grep "❌\|ERROR\|Exception" auto_batch.log
```

## 🎯 성공 지표

### 완전 자동화 성공 기준
1. ✅ GUI 창이 열리지 않음
2. ✅ 사용자 입력 없이 자동 진행
3. ✅ 모든 세트가 순차적으로 완료
4. ✅ 각 단계별 결과물 자동 생성

### 품질 지표
- **마스크 생성 성공률**: 90% 이상
- **세그멘테이션 정확도**: YOLO 감지 품질에 의존
- **처리 시간**: GUI 버전 대비 20-30% 단축 예상

## 🚀 다음 단계

### 단기 개선사항
1. **마스크 품질 향상**: 더 정교한 객체 경계 감지
2. **에러 처리 강화**: 다양한 실패 케이스 대응
3. **진행률 표시**: 더 상세한 진행 상황 모니터링

### 장기 개선사항
1. **적응형 학습**: 데이터셋 특성에 따른 자동 파라미터 조정
2. **병렬 처리**: 여러 세트 동시 실행 (리소스 허용시)
3. **웹 대시보드**: 실행 상태를 웹에서 모니터링

---

## 🎉 결론

**이제 진정한 의미의 자동화가 완성되었습니다!**

- ❌ **기존**: GUI 창 열림 → 사용자 클릭/입력 → 수동 인터럽트
- ✅ **현재**: 명령행 실행 → YOLO 자동 마스크 생성 → 완전 자동 완료

**사용법**: `python batch_automation_auto.py` 한 번만 실행하면 모든 세트가 GUI 없이 자동으로 완료됩니다!
