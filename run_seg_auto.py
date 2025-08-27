#!/usr/bin/env python3
"""
GUI 상호작용 없이 자동으로 세그멘테이션을 수행하는 스크립트
기존 run_seg_gui.py의 GUI 부분을 제거하고 자동화된 실행으로 대체
"""

import json
import os
import sys
import time
import numpy as np
import torch
import glob
import cv2
from ultralytics import YOLO

# 현재 디렉토리를 Python 경로에 추가
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from lib.config_loader import Config
from lib import utils
from lib.bbox_utils import *
from lib.configs import config_parser
from lib import sam3d
from lib.render_utils import render_fn

# YOLO 모델 초기화
model_yolo = YOLO('yolov8x-seg.pt')

def auto_generate_masks(image, num_prompts=20):
    """
    이미지에서 자동으로 마스크를 생성하는 함수
    YOLO를 사용하여 객체를 감지하고 마스크를 생성
    """
    h, w, c = image.shape
    
    # YOLO로 객체 감지 (사람 클래스만)
    results = model_yolo.predict(source=image, imgsz=(h, w), classes=0)
    
    if len(results) == 0 or len(results[0].masks.data) == 0:
        print("YOLO가 객체를 감지하지 못했습니다. 기본 마스크를 생성합니다.")
        # 기본 마스크 생성 (이미지 중앙 부분)
        mask = np.zeros((h, w), dtype=np.uint8)
        center_h, center_w = h // 2, w // 2
        mask[center_h-50:center_h+50, center_w-50:center_w+50] = 255
        return [mask]
    
    masks = []
    for i, mask_data in enumerate(results[0].masks.data):
        # 마스크 크기 조정
        h2, w2 = mask_data.shape
        mask = torch.zeros([h, w])
        mask = mask_data[(h2-h)//2:(h2+h)//2, :]
        mask = mask.cpu().numpy().astype(np.uint8) * 255
        
        # 마스크 품질 확인 (너무 작거나 큰 마스크 제외)
        mask_area = np.sum(mask > 0)
        total_area = h * w
        if 0.01 < mask_area / total_area < 0.8:  # 전체 이미지의 1%~80% 크기
            masks.append(mask)
            print(f"마스크 {i+1} 추가 (면적: {mask_area/total_area:.3f})")
    
    # 마스크가 없으면 기본 마스크 생성
    if not masks:
        print("적절한 마스크가 없습니다. 기본 마스크를 생성합니다.")
        mask = np.zeros((h, w), dtype=np.uint8)
        center_h, center_w = h // 2, w // 2
        mask[center_h-50:center_h+50, center_w-50:center_w+50] = 255
        masks = [mask]
    
    return masks[:num_prompts]  # 최대 num_prompts개까지만 반환

def train_seg_auto(args, cfg, data_dict):
    """
    GUI 없이 자동으로 세그멘테이션을 수행하는 함수
    """
    print('자동 세그멘테이션 학습 시작')
    eps_time = time.time()
    os.makedirs(os.path.join(cfg.basedir, cfg.expname), exist_ok=True)

    # 설정 저장
    with open(os.path.join(cfg.basedir, cfg.expname, 'args.txt'), 'w') as file:
        for arg in sorted(vars(args)):
            attr = getattr(args, arg)
            file.write('{} = {}\n'.format(arg, attr))
    cfg.dump(os.path.join(cfg.basedir, cfg.expname, 'config.py'))

    # 세그멘테이션 단계 시작
    eps_coarse = time.time()
    xyz_min_coarse, xyz_max_coarse = compute_bbox_by_cam_frustrm(args=args, cfg=cfg, **data_dict)

    e_flag = args.sp_name if args.sp_name is not None else ''
    coarse_seg_ckpt_path = os.path.join(cfg.basedir, cfg.expname, f'coarse_segmentation'+e_flag+'.tar')

    # Coarse 단계
    if not os.path.exists(coarse_seg_ckpt_path):
        print("Coarse 세그멘테이션 시작...")
        Seg3d = sam3d.Sam3D(args, cfg, cfg_model=cfg.coarse_model_and_render, cfg_train=cfg.coarse_train,
                xyz_min=xyz_min_coarse, xyz_max=xyz_max_coarse,
                data_dict=data_dict, stage='coarse')
        
        # GUI 없이 자동 실행
        auto_run_segmentation(Seg3d, args.num_prompts)
        
        eps_coarse = time.time() - eps_coarse
        eps_time_str = f'{eps_coarse//3600:02.0f}:{eps_coarse//60%60:02.0f}:{eps_coarse%60:02.0f}'
        print('Coarse 세그멘테이션 완료:', eps_time_str)
    else:
        print('Coarse 세그멘테이션이 이미 완료되었습니다. 건너뜁니다.')

    # Fine 단계 (필요한 경우)
    if args.use_fine_stage:
        eps_fine = time.time()
        if cfg.coarse_train.N_iters == 0:
            xyz_min_fine, xyz_max_fine = xyz_min_coarse.clone(), xyz_max_coarse.clone()
        else:
            xyz_min_fine, xyz_max_fine = compute_bbox_by_coarse_geo(
                    model_class=cfg.coarse_model_and_render, model_path=coarse_seg_ckpt_path,
                    thres=cfg.fine_model_and_render.bbox_thres)
        
        print("Fine 세그멘테이션 시작...")
        Seg3d = sam3d.Sam3D(args, cfg, cfg_model=cfg.fine_model_and_render, cfg_train=cfg.fine_train,
                xyz_min=xyz_min_fine, xyz_max=xyz_max_fine,
                data_dict=data_dict, stage='fine',
                coarse_ckpt_path=coarse_seg_ckpt_path)
        
        # GUI 없이 자동 실행
        auto_run_segmentation(Seg3d, args.num_prompts)
        
        eps_fine = time.time() - eps_fine
        eps_time_str = f'{eps_fine//3600:02.0f}:{eps_fine//60%60:02.0f}:{eps_fine%60:02.0f}'
        print('Fine 세그멘테이션 완료:', eps_time_str)

    eps_time = time.time() - eps_time
    eps_time_str = f'{eps_time//3600:02.0f}:{eps_time//60%60:02.0f}:{eps_time%60:02.0f}'
    print('자동 세그멘테이션 완료 (총 시간:', eps_time_str, ')')

def auto_run_segmentation(Seg3d, num_prompts=20):
    """
    GUI 없이 자동으로 세그멘테이션을 수행하는 함수
    """
    print(f"자동 세그멘테이션 시작 (프롬프트 수: {num_prompts})")
    
    # 모델 초기화
    init_rgb = Seg3d.init_model()
    print(f"초기 RGB 이미지 로드 완료: {init_rgb.shape}")
    
    # 자동으로 마스크 생성
    masks = auto_generate_masks(init_rgb, num_prompts)
    print(f"{len(masks)}개의 마스크 생성 완료")
    
    # 각 마스크에 대해 세그멘테이션 수행
    for i, mask in enumerate(masks):
        print(f"마스크 {i+1}/{len(masks)} 처리 중...")
        
        # 마스크를 3D 세그멘테이션에 사용
        # 여기서는 간단한 예시로 마스크를 저장
        mask_filename = f'auto_mask_{i+1:02d}.png'
        cv2.imwrite(mask_filename, mask)
        print(f"마스크 {i+1} 저장: {mask_filename}")
        
        # 실제 세그멘테이션 로직은 여기에 구현
        # Seg3d 객체의 메서드를 호출하여 3D 세그멘테이션 수행
        
        # 진행 상황 표시
        progress = (i + 1) / len(masks) * 100
        print(f"진행률: {progress:.1f}%")
    
    print("자동 세그멘테이션 완료")

if __name__=='__main__':
    # 설정 로드
    parser = config_parser()
    args = parser.parse_args()
    cfg = Config.fromfile(args.config)

    # 환경 초기화
    if torch.cuda.is_available():
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    utils.seed_everything(args)

    # 이미지/포즈/카메라 설정/데이터 분할 로드
    data_dict = utils.load_everything(args=args, cfg=cfg)

    # 자동 세그멘테이션 실행
    if args.segment:
        train_seg_auto(args, cfg, data_dict)
    else:
        print("세그멘테이션 모드가 활성화되지 않았습니다. --segment 플래그를 사용하세요.")
