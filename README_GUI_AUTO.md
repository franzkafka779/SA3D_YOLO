# SA3D YOLO GUI 상호작용 자동화 가이드

이 문서는 SA3D YOLO의 GUI 상호작용을 자동화하는 방법을 설명합니다.

## 🚀 문제 상황

기존 `run_seg_gui.py`를 실행할 때 사용자가 GUI에서 수동으로 해야 하는 상호작용들:

1. **Prompt Type을 'text'로 설정**
2. **Text Prompt를 'none'으로 설정하고 Generate 버튼 클릭**
3. **Please select mask에서 아무 마스크나 선택 (모두 같음)**
4. **SA3D Training 부분에서 Start Training 버튼 클릭**

이 과정들이 자동화되지 않아 배치 처리에 어려움이 있었습니다.

## 💡 해결 방안

### 1. 자동화된 GUI 클래스 생성

`AutoInteractionSam3dGUI` 클래스를 만들어서 사용자 상호작용을 자동으로 수행합니다:

```python
class AutoInteractionSam3dGUI(Sam3dGUI):
    def perform_auto_interaction(self, init_rgb):
        # 1. Prompt type을 text로 설정
        self.ctx['prompt_type'] = 'text'
        
        # 2. Text Prompt를 "none"으로 설정하고 Generate 버튼 클릭
        text_prompt = "none"
        masks, fig0, fig1, fig2, fig3 = query(points=None, text=text_prompt)
        
        # 3. 마스크 선택 (첫 번째 마스크)
        self.ctx['select_mask_id'] = 0
        
        # 4. Start Training 버튼 클릭
        self.start_auto_training(selected_mask)
```

### 2. 자동화된 스크립트

`run_seg_gui_auto_interaction.py`는 기존 `run_seg_gui.py`의 기능을 유지하면서 GUI 상호작용을 자동화합니다.

## 🔧 사용법

### 1. 2번째 셀만 자동화 (권장)

```bash
# 2번째 셀 자동화 실행
python batch_seg_only_auto.py
```

**사전 조건**: 1번째 셀(세그멘테이션 학습)이 완료되어야 합니다.

### 2. 개별 세트 렌더링 자동화

```bash
# Set6만 자동 렌더링
python run_seg_gui_auto_interaction.py --config=configs/nerf_unbounded/seg_Set6.py --segment --sp_name=_gui --num_prompts=20 --render_only --render_opt=video --dump_images --seg_type seg_img seg_density
```

### 3. 세그멘테이션 학습 자동화

```bash
# 세그멘테이션 학습 자동화 (GUI 상호작용 없음)
python run_seg_gui_auto_interaction.py --config=configs/nerf_unbounded/seg_Set6.py --segment --sp_name=_gui --num_prompts=20 --render_opt=train --save_ckpt
```

## 📋 자동화 과정

### 1단계: Prompt Type 설정
- 자동으로 `'text'`로 설정
- 사용자 입력 불필요

### 2단계: Text Prompt 생성
- 자동으로 `'none'`으로 설정
- Generate 버튼 자동 클릭
- YOLO 기반 마스크 자동 생성

### 3단계: 마스크 선택
- 자동으로 첫 번째 마스크(mask0) 선택
- 모든 마스크가 동일하므로 선택 순서 무관

### 4단계: 훈련 시작
- Start Training 버튼 자동 클릭
- 훈련 과정 자동 진행
- 진행 상황 실시간 모니터링

## ⚠️ 주의사항

### 1. 사전 조건
- **1번째 셀 완료 필수**: 세그멘테이션 학습이 완료되어야 함
- **체크포인트 파일 존재**: `coarse_segmentation_gui.tar` 또는 `fine_segmentation_gui.tar`

### 2. 제한사항
- **마스크 품질**: YOLO 감지 품질에 의존
- **훈련 시간**: 데이터셋 크기에 따라 달라짐
- **메모리 사용량**: GPU 메모리 모니터링 필요

### 3. 오류 처리
- 자동화 실패 시 수동 GUI 모드로 전환
- 상세한 로그 출력으로 문제 진단 가능

## 🔍 문제 해결

### 일반적인 문제들

**1. 마스크 생성 실패**
```bash
# YOLO 모델 확인
ls -la yolov8x-seg.pt

# GPU 메모리 확인
nvidia-smi
```

**2. 훈련 중단**
```bash
# 로그 확인
tail -f logs/segmentation.log

# 체크포인트 파일 확인
ls -la logs/coarse_segmentation_gui.tar
```

**3. 렌더링 실패**
```bash
# 설정 파일 확인
cat configs/nerf_unbounded/seg_Set6.py

# 데이터 경로 확인
ls -la data/
```

## 📊 모니터링

### 진행 상황 확인
```bash
# 실시간 로그 모니터링
tail -f auto_interaction.log

# 특정 단계 확인
grep "자동 상호작용" auto_interaction.log
grep "훈련 진행 중" auto_interaction.log
grep "렌더링 완료" auto_interaction.log
```

### 성공 지표
- ✅ Prompt type 자동 설정
- ✅ Text prompt 자동 생성
- ✅ 마스크 자동 선택
- ✅ 훈련 자동 시작
- ✅ 렌더링 자동 완료

## 🚀 다음 단계

### 단기 개선사항
1. **마스크 품질 향상**: 더 정교한 객체 감지
2. **훈련 진행률 표시**: 상세한 진행 상황 모니터링
3. **오류 복구**: 자동 재시도 메커니즘

### 장기 개선사항
1. **적응형 파라미터**: 데이터셋 특성에 따른 자동 조정
2. **병렬 처리**: 여러 세트 동시 실행
3. **웹 대시보드**: 실행 상태 시각화

## 🎉 결론

**이제 GUI 상호작용 없이도 세그멘테이션 학습과 렌더링을 자동으로 수행할 수 있습니다!**

- ❌ **기존**: GUI 창 열림 → 수동 상호작용 → 시간 소모
- ✅ **현재**: 명령행 실행 → 자동 상호작용 → 완전 자동화

**사용법**: `python batch_seg_only_auto.py` 한 번만 실행하면 모든 세트의 렌더링이 GUI 상호작용 없이 자동으로 완료됩니다!
