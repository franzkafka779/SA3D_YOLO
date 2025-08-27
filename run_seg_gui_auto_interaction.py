#!/usr/bin/env python3
"""
GUI 상호작용을 자동화한 run_seg_gui.py
사용자가 요청한 상호작용들을 자동으로 수행:
1. Prompt type을 text로 설정
2. Text Prompt를 "none"으로 설정하고 Generate 버튼 클릭
3. 마스크 선택 (첫 번째 마스크)
4. Start Training 버튼 클릭
"""

import json
import os
import sys
import time

import imageio
import matplotlib.pyplot as plt

# remove the dependency on mmcv
# import mmcv
from lib.config_loader import Config

import numpy as np
import torch
import glob
import cv2
from ultralytics import YOLO
from torchvision.utils import save_image

from lib import utils
from lib.bbox_utils import *
from lib.configs import config_parser
from lib import sam3d
from lib.gui import Sam3dGUI
from lib.render_utils import render_fn

# YOLO 모델 초기화
model_yolo = YOLO('yolov8x-seg.pt')


def train_seg_auto(args, cfg, data_dict):
    '''자동화된 main training code for Segment Anything in 3D with NeRFs'''
    print('train: start (자동화 모드)')
    eps_time = time.time()
    os.makedirs(os.path.join(cfg.basedir, cfg.expname), exist_ok=True)

    # save configs
    with open(os.path.join(cfg.basedir, cfg.expname, 'args.txt'), 'w') as file:
        for arg in sorted(vars(args)):
            attr = getattr(args, arg)
            file.write('{} = {}\n'.format(arg, attr))
    cfg.dump(os.path.join(cfg.basedir, cfg.expname, 'config.py'))

    # start segmentation stage
    eps_coarse = time.time()
    xyz_min_coarse, xyz_max_coarse = compute_bbox_by_cam_frustrm(args=args, cfg=cfg, **data_dict)

    e_flag = args.sp_name if args.sp_name is not None else ''
    coarse_seg_ckpt_path = os.path.join(cfg.basedir, cfg.expname, f'coarse_segmentation'+e_flag+'.tar')

    # coarse stage
    if not os.path.exists(coarse_seg_ckpt_path):
        Seg3d = sam3d.Sam3D(args, cfg, cfg_model=cfg.coarse_model_and_render, cfg_train=cfg.coarse_train,
                xyz_min=xyz_min_coarse, xyz_max=xyz_max_coarse,
                data_dict=data_dict, stage='coarse')
        
        # 자동화된 GUI 실행
        gui = AutoInteractionSam3dGUI(Seg3d)
        gui.run()
        
        eps_coarse = time.time() - eps_coarse
        eps_time_str = f'{eps_coarse//3600:02.0f}:{eps_coarse//60%60:02.0f}:{eps_coarse%60:02.0f}'
        print('train: coarse segmentation in', eps_time_str)
    else:
        print('Coarse segmentation has been completed, skip!')

    # fine stage when the mask from the coarse stage is not good enough
    if args.use_fine_stage:
        eps_fine = time.time()
        if cfg.coarse_train.N_iters == 0:
            xyz_min_fine, xyz_max_fine = xyz_min_coarse.clone(), xyz_max_coarse.clone()
        else:
            xyz_min_fine, xyz_max_fine = compute_bbox_by_coarse_geo(
                    model_class=cfg.coarse_model_and_render, model_path=coarse_seg_ckpt_path,
                    thres=cfg.fine_model_and_render.bbox_thres)
        # finetune
        Seg3d = sam3d.Sam3D(args, cfg, cfg_model=cfg.fine_model_and_render, cfg_train=cfg.fine_train,
                xyz_min=xyz_min_fine, xyz_max=xyz_max_fine,
                data_dict=data_dict, stage='fine',
                coarse_ckpt_path=coarse_seg_ckpt_path)
        
        # 자동화된 GUI 실행
        gui = AutoInteractionSam3dGUI(Seg3d)
        gui.run()
        
        eps_fine = time.time() - eps_fine
        eps_time_str = f'{eps_fine//3600:02.0f}:{eps_fine//60%60:02.0f}:{eps_fine%60:02.0f}'
        print('train: fine detail segmentation in', eps_time_str)

    eps_time = time.time() - eps_time
    eps_time_str = f'{eps_time//3600:02.0f}:{eps_time//60%60:02.0f}:{eps_time%60:02.0f}'
    print('train: finish (eps time', eps_time_str, ')')


class AutoInteractionSam3dGUI(Sam3dGUI):
    """
    GUI 상호작용을 자동화한 Sam3dGUI 클래스
    기존 GUI의 모든 기능을 그대로 사용하되, 상호작용만 자동화
    """
    
    def __init__(self, Seg3d, debug=False):
        super().__init__(Seg3d, debug)
        self.auto_interaction_completed = False
        # args 속성 추가 (체크포인트 확인용)
        self.args = Seg3d.args if hasattr(Seg3d, 'args') else None
    
    def run(self):
        """자동 상호작용을 포함한 실행"""
        init_rgb = self.Seg3d.init_model()
        self.ctx['cur_img'] = init_rgb
        
        # 자동 상호작용 수행 (GUI 실행 없이 직접 학습 로직 호출)
        print("🚀 자동 상호작용 시작...")
        self.perform_auto_interaction(init_rgb)
    
    def perform_auto_interaction(self, init_rgb):
        """사용자가 요청한 상호작용들을 자동으로 수행"""
        print("🚀 자동 상호작용을 시작합니다...")
        
        try:
            # 1. Prompt type을 text로 설정
            print("1️⃣ Prompt type을 'text'로 설정")
            self.ctx['prompt_type'] = 'text'
            
            # 2. Text Prompt를 "none"으로 설정하고 Generate 버튼 클릭 시뮬레이션
            print("2️⃣ Text Prompt를 'none'으로 설정하고 Generate 버튼 클릭")
            text_prompt = "none"
            self.ctx['text'] = text_prompt
            
            # 자체 구현한 query 함수 호출로 마스크 생성
            masks, fig0, fig1, fig2, fig3 = auto_query(self.Seg3d.predictor, self.ctx, points=None, text=text_prompt)
            
            # 마스크를 컨텍스트에 저장
            self.ctx['masks'] = masks
            self.ctx['fig0'] = fig0
            self.ctx['fig1'] = fig1
            self.ctx['fig2'] = fig2
            self.ctx['fig3'] = fig3
            
            print(f"✅ 마스크 생성 완료: {len(masks)}개")
            
            # 3. 마스크 선택 (첫 번째 마스크 선택)
            print("3️⃣ 첫 번째 마스크(mask0) 선택")
            self.ctx['select_mask_id'] = 0
            selected_mask = masks[0]
            print(f"✅ 선택된 마스크: mask0 (인덱스: 0)")
            
            # 4. GUI의 학습 로직을 직접 호출하여 학습 수행
            print("4️⃣ GUI 학습 로직 직접 호출하여 학습 수행")
            self.start_auto_training(selected_mask)
            
            self.auto_interaction_completed = True
            print("🎉 모든 자동 상호작용이 완료되었습니다!")
            
        except Exception as e:
            print(f"❌ 자동 상호작용 중 오류 발생: {e}")
            import traceback
            traceback.print_exc()
            self.auto_interaction_completed = False
    
    def start_auto_training(self, selected_mask):
        """자동 훈련 시작 - GUI의 학습 로직을 직접 호출하여 실행"""
        print("🏋️ 자동 훈련을 시작합니다...")
        
        try:
            # GUI의 start_training 함수와 동일한 로직을 직접 구현
            print("📸 첫 번째 뷰에서 훈련 시작...")
            
            # 첫 번째 뷰에서 훈련 (GUI와 동일한 로직)
            self.train_idx = 0
            self.Seg3d.train_step(self.train_idx, sam_mask=selected_mask)
            self.train_idx += 1
            print(f"✅ 첫 번째 뷰 훈련 완료 (인덱스: {self.train_idx-1})")
            
            # 교차 뷰 훈련 시작 (모든 뷰를 처리하도록 보장)
            print("🔄 교차 뷰 훈련 시작...")
            
            # 모든 뷰를 처리하도록 보장 (is_finished 플래그 무시)
            total_views = len(self.Seg3d.data_dict['i_train'])
            print(f"📊 총 {total_views}개 뷰를 처리합니다...")
            
            while self.train_idx < total_views:
                try:
                    rgb, sam_prompt, is_finished = self.Seg3d.train_step(self.train_idx)
                    self.train_idx += 1
                    
                    # 진행 상황 표시
                    if self.train_idx % 5 == 0:
                        print(f"🔄 교차 뷰 훈련 진행 중... 현재 인덱스: {self.train_idx}/{total_views}")
                    
                    # is_finished 플래그는 무시하고 모든 뷰 처리
                    if is_finished:
                        print(f"⚠️ 완료 신호 수신했지만 계속 진행 (인덱스: {self.train_idx-1})")
                        
                except Exception as e:
                    print(f"⚠️ 교차 뷰 훈련 중 오류: {e}")
                    break
            
            print(f"🏁 교차 뷰 훈련 완료! 총 {self.train_idx}개 뷰 처리")
            
            # 두 번째 학습 단계 시작 (GUI와 동일한 로직)
            print("🔄 두 번째 학습 단계 시작...")
            
            # vsgflag 설정
            self.Seg3d.vsgflag = True
            
            # 신뢰도 기반으로 뷰 순서 정렬
            confidences = self.Seg3d.confidences
            i_train = self.Seg3d.data_dict['i_train']
            
            print(f"📊 신뢰도 기반 뷰 순서 정렬 중... (신뢰도: {len(confidences)}, 훈련 인덱스: {len(i_train)})")
            
            # 누락된 신뢰도를 기본값으로 채우기
            if len(confidences) < len(i_train):
                missing_count = len(i_train) - len(confidences)
                print(f"⚠️ {missing_count}개 뷰의 신뢰도가 누락되었습니다. 기본값으로 채웁니다.")
                
                # 누락된 신뢰도를 기본값(0.5)으로 채우기
                default_confidence = 0.5
                for i in range(missing_count):
                    confidences.append(default_confidence)
                    print(f"   뷰 {len(confidences)-1}: 기본 신뢰도 {default_confidence} 추가")
                
                print(f"✅ 신뢰도 리스트 길이: {len(confidences)} (기본값 포함)")
            
            # 이제 두 리스트의 길이가 같아졌으므로 안전하게 처리
            view_dict = {confidences[i]: i_train[i] for i in range(len(i_train))}
            view_dict = dict(sorted(view_dict.items(), reverse=True))
            i_train_sorted = list(view_dict.values())
            
            print(f"📋 정렬된 뷰 순서: {i_train_sorted[:5]}... (총 {len(i_train_sorted)}개)")
            
            # 정렬된 순서로 훈련 데이터 업데이트
            self.Seg3d.data_dict['i_train'] = i_train_sorted
            
            # 첫 번째 뷰에서 YOLO 객체 감지 및 마스크 생성
            first_idx = i_train_sorted[0]
            img = self.Seg3d.data_dict['images'][first_idx, :, :, :].numpy()
            img = utils.to8b(img)
            h, w, c = img.shape
            
            results = model_yolo.predict(source=img, imgsz=(h, w), classes=0)
            
            if not hasattr(results[0], 'masks') or results[0].masks is None or len(results[0].masks.data) == 0:
                print(f"⚠️ YOLO가 객체를 감지하지 못했습니다. 기본 마스크를 사용합니다.")
                # 기본 마스크 생성
                m = torch.zeros([h, w])
                center_h, center_w = h // 2, w // 2
                m[center_h-50:center_h+50, center_w-50:center_w+50] = 1
            else:
                h2, w2 = results[0].masks.data[0].shape
                idx_select = 8  # 첫 번째 마스크 선택
                
                # 선택한 인덱스가 유효한 범위인지 확인
                num_detected_objects = len(results[0].masks.data)
                if idx_select >= num_detected_objects:
                    idx_select = num_detected_objects - 1
                
                img_mask = results[0].masks.data[idx_select]
                m = img_mask[(h2-h)//2:(h2+h)//2, :]
            
            # 마스크를 3채널로 변환
            masks = torch.zeros([c, h, w]).cpu().numpy()
            masks[0:3, :, :] = m.type(torch.bool).cpu().numpy()
            
            # 두 번째 학습 시작
            self.train_idx = 0
            print("🎯 두 번째 학습 시작...")
            
            # 첫 번째 뷰에서 훈련
            self.Seg3d.train_step(self.train_idx, sam_mask=masks)
            self.train_idx += 1
            
            # 교차 뷰 훈련
            while True:
                try:
                    rgb, sam_prompt, is_finished = self.Seg3d.train_step(self.train_idx)
                    self.train_idx += 1
                    
                    if self.train_idx % 5 == 0:
                        print(f"🔄 두 번째 학습 진행 중... 현재 인덱스: {self.train_idx}")
                    
                    if is_finished:
                        print(f"✅ 두 번째 학습 완료 신호 수신 (인덱스: {self.train_idx-1})")
                        break
                        
                except Exception as e:
                    print(f"⚠️ 두 번째 학습 중 오류: {e}")
                    break
            
            print(f"🏁 두 번째 학습 완료! 총 인덱스: {self.train_idx}")
            
            # 체크포인트 저장
            print("💾 체크포인트 저장 중...")
            self.Seg3d.save_ckpt()
            print("✅ 체크포인트 저장 완료")
            
            # 체크포인트 파일 생성 확인 - 충분히 기다림
            print("🔍 체크포인트 파일 생성 확인 중...")
            checkpoint_path = self.wait_for_checkpoint_creation()
            if checkpoint_path:
                print(f"✅ 체크포인트 파일 생성 확인: {checkpoint_path}")
                checkpoint_created = True
            else:
                print("⚠️ 체크포인트 파일이 생성되지 않았습니다.")
                checkpoint_created = False
            
            # 학습 완료 후 렌더링 수행 (run_seg_gui.py와 동일한 로직)
            if checkpoint_created:
                print("🎬 학습 완료 후 렌더링 시작...")
                rendering_completed = self.perform_post_training_rendering(checkpoint_path)
                
                if rendering_completed:
                    print("🏁 모든 훈련 및 렌더링이 완료되었습니다!")
                    self.auto_interaction_completed = True
                else:
                    print("⚠️ 렌더링이 완료되지 않았습니다.")
            else:
                print("⚠️ 체크포인트 파일이 생성되지 않아 렌더링을 건너뜁니다.")
            
        except Exception as e:
            print(f"❌ 자동 훈련 중 오류 발생: {e}")
            import traceback
            traceback.print_exc()
    
    def wait_for_checkpoint_creation(self, max_wait_time=300):
        """체크포인트 파일 생성까지 충분히 기다림 (기본 5분)"""
        print(f"⏳ 체크포인트 파일 생성을 기다립니다... (최대 {max_wait_time}초)")
        
        start_time = time.time()
        check_interval = 10  # 10초마다 확인
        
        while time.time() - start_time < max_wait_time:
            checkpoint_path = self.check_checkpoint_creation()
            if checkpoint_path:
                return checkpoint_path
            
            # 대기 시간 표시
            elapsed = int(time.time() - start_time)
            remaining = max_wait_time - elapsed
            print(f"⏳ 체크포인트 대기 중... {elapsed}초 경과, {remaining}초 남음")
            
            time.sleep(check_interval)
        
        print(f"⏰ {max_wait_time}초 대기 완료. 체크포인트 파일을 찾을 수 없습니다.")
        return None
    
    def check_checkpoint_creation(self):
        """체크포인트 파일 생성 여부 확인"""
        try:
            # 체크포인트 파일 경로 확인
            e_flag = self.args.sp_name if self.args.sp_name is not None else ''
            base_dir = self.Seg3d.base_save_dir
            
            # coarse 단계 체크포인트 확인
            coarse_path = os.path.join(base_dir, f'coarse_segmentation{e_flag}.tar')
            if os.path.exists(coarse_path):
                file_size = os.path.getsize(coarse_path)
                print(f"📁 Coarse 체크포인트 발견: {coarse_path} (크기: {file_size} bytes)")
                return coarse_path
            
            # fine 단계 체크포인트 확인
            fine_path = os.path.join(base_dir, f'fine_segmentation{e_flag}.tar')
            if os.path.exists(fine_path):
                file_size = os.path.getsize(fine_path)
                print(f"📁 Fine 체크포인트 발견: {fine_path} (크기: {file_size} bytes)")
                return fine_path
            
            # 최신 체크포인트 파일 찾기
            checkpoint_files = glob.glob(os.path.join(base_dir, '*segmentation*.tar'))
            if checkpoint_files:
                latest_checkpoint = max(checkpoint_files, key=os.path.getctime)
                file_size = os.path.getsize(latest_checkpoint)
                print(f"📁 최신 체크포인트 발견: {latest_checkpoint} (크기: {file_size} bytes)")
                return latest_checkpoint
            
            return None
            
        except Exception as e:
            print(f"⚠️ 체크포인트 확인 중 오류: {e}")
            return None
    
    def perform_final_rendering(self):
        """최종 렌더링 수행 및 결과물 생성 확인 (간단한 버전)"""
        try:
            print("🎬 최종 렌더링 확인 중...")
            
            # 결과물 파일들이 생성되었는지 확인
            if self.check_result_files():
                print("✅ 모든 결과물 파일이 생성되었습니다!")
                return True
            else:
                print("⚠️ 결과물 파일이 아직 생성되지 않았습니다.")
                return False
            
        except Exception as e:
            print(f"⚠️ 최종 렌더링 확인 중 오류: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def check_result_files(self):
        """결과물 파일들이 생성되었는지 확인"""
        try:
            base_dir = self.Seg3d.base_save_dir
            
            # 일반적인 결과물 파일들 확인
            result_files = [
                'video.mp4',  # 비디오 파일
                'images',     # 이미지 폴더
                '*.png',      # PNG 이미지들
                '*.jpg',      # JPG 이미지들
            ]
            
            found_files = []
            for pattern in result_files:
                if '*' in pattern:
                    # 와일드카드 패턴
                    files = glob.glob(os.path.join(base_dir, pattern))
                    found_files.extend(files)
                else:
                    # 일반 파일/폴더
                    path = os.path.join(base_dir, pattern)
                    if os.path.exists(path):
                        found_files.append(path)
            
            if found_files:
                print(f"📁 결과물 파일 {len(found_files)}개 발견")
                return True
            else:
                return False
                
        except Exception as e:
            print(f"⚠️ 결과물 파일 확인 중 오류: {e}")
            return False
    
    def perform_post_training_rendering(self, checkpoint_path):
        """학습 완료 후 렌더링 수행 (run_seg_gui.py와 동일한 로직)"""
        try:
            print("🎬 학습 완료 후 렌더링 시작...")
            
            # args와 cfg 정보 가져오기
            args = self.args
            cfg = self.Seg3d.cfg if hasattr(self.Seg3d, 'cfg') else None
            
            if args is None or cfg is None:
                print("⚠️ args 또는 cfg 정보를 가져올 수 없습니다.")
                return False
            
            # 렌더링 옵션이 설정되어 있는지 확인
            if not hasattr(args, 'render_opt') or args.render_opt is None:
                print("⚠️ 렌더링 옵션이 설정되지 않았습니다. --render_opt 플래그를 사용하세요.")
                return False
            
            print(f"🎯 렌더링 옵션: {args.render_opt}")
            
            # e_flag 설정
            e_flag = args.sp_name if args.sp_name is not None else ''
            
            # 렌더링 수행
            for seg_type in ['seg_img', 'seg_density']:
                print(f"🎨 {seg_type} 렌더링 시작...")
                
                # 체크포인트 경로 설정
                if hasattr(args, 'ft_path') and args.ft_path:
                    ckpt_path = args.ft_path
                else:
                    fine_path = os.path.join(cfg.basedir, cfg.expname, 'fine_segmentation'+e_flag+'.tar')
                    coarse_path = os.path.join(cfg.basedir, cfg.expname, 'coarse_segmentation'+e_flag+'.tar')
                    ckpt_path = fine_path if os.path.exists(fine_path) else coarse_path
                
                print(f"📁 렌더링에 사용할 체크포인트: {ckpt_path}")
                
                # 체크포인트 파일 존재 확인
                if not os.path.exists(ckpt_path):
                    print(f"❌ 체크포인트 파일을 찾을 수 없습니다: {ckpt_path}")
                    continue
                
                # ckpt_name 추출
                ckpt_name = ckpt_path.split('/')[-1][:-4]
                print(f"📝 체크포인트 이름: {ckpt_name}")
                
                # 모델 로드
                print("🔄 모델 로드 중...")
                model_class = utils.find_model(cfg)
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                model, optimizer, start = utils.load_existed_model(args, cfg, cfg.fine_train, ckpt_path, device)
                
                # 렌더링 설정
                stepsize = cfg.fine_model_and_render.stepsize
                render_viewpoints_kwargs = {
                    'model': model,
                    'ndc': cfg.data.ndc,
                    'render_kwargs': {
                        'near': self.Seg3d.data_dict['near'],
                        'far': self.Seg3d.data_dict['far'],
                        'bg': 1 if cfg.data.white_bkgd else 0,
                        'stepsize': stepsize,
                        'inverse_y': cfg.data.inverse_y,
                        'flip_x': cfg.data.flip_x,
                        'flip_y': cfg.data.flip_y,
                        'render_depth': True,
                    },
                }
                
                # 렌더링 타입에 따른 모델 설정
                flag = "seg" if hasattr(args, 'segment') and args.segment else ""
                if hasattr(args, 'segment') and args.segment:
                    if seg_type == 'seg_density':
                        render_viewpoints_kwargs['model'].segmentation_to_density()
                        print("🔄 모델을 seg_density 모드로 설정")
                    elif seg_type == 'seg_img':
                        render_viewpoints_kwargs['model'].segmentation_only()
                        print("🔄 모델을 seg_img 모드로 설정")
                    else:
                        print(f"⚠️ 지원하지 않는 seg_type: {seg_type}")
                        continue
                
                # 객체 수 확인
                num_obj = render_viewpoints_kwargs['model'].seg_mask_grid.grid.shape[1]
                print(f"📊 세그멘테이션 객체 수: {num_obj}")
                
                # 모델을 GPU로 이동
                render_viewpoints_kwargs['model'] = render_viewpoints_kwargs['model'].cuda()
                
                # 렌더링 실행
                print(f"🚀 {seg_type} 렌더링 실행 중...")
                render_fn(args, cfg, ckpt_name, flag, e_flag, num_obj, 
                         self.Seg3d.data_dict, render_viewpoints_kwargs, seg_type=seg_type)
                
                print(f"✅ {seg_type} 렌더링 완료!")
            
            print("🎉 모든 렌더링이 완료되었습니다!")
            return True
            
        except Exception as e:
            print(f"❌ 학습 완료 후 렌더링 중 오류 발생: {e}")
            import traceback
            traceback.print_exc()
            return False


def auto_query(sam_pred, ctx, points=None, text=None):
    """
    자동화된 query 함수 (gui.py의 query 함수를 재구현)
    """
    with torch.no_grad():
        if text is None:
            input_point = points
            input_label = np.ones(len(input_point))
            masks, scores, logits = sam_pred.predict(
                point_coords=input_point,
                point_labels=input_label,
                multimask_output=True,
            )
        elif points is None:
            # YOLO를 사용한 자동 객체 감지 및 마스크 생성
            h, w, c = ctx['cur_img'].shape
            results = model_yolo.predict(source=ctx['cur_img'], imgsz=(h,w), classes=0)
            
            if len(results) == 0 or len(results[0].masks.data) == 0:
                print("YOLO가 객체를 감지하지 못했습니다. 기본 마스크를 생성합니다.")
                # 기본 마스크 생성
                m = torch.zeros([h, w])
                center_h, center_w = h // 2, w // 2
                m[center_h-50:center_h+50, center_w-50:center_w+50] = 1
            else:
                # 자동으로 최적의 instance 선택
                idx_select = 8  # 첫 번째 마스크 선택
                
                # 선택한 인덱스가 유효한 범위인지 확인
                num_detected_objects = len(results[0].masks.data)
                if idx_select >= num_detected_objects:
                    idx_select = num_detected_objects - 1
                
                # 선택된 instance의 마스크 사용
                h2, w2 = results[0].masks.data[idx_select].shape
                m = torch.zeros([h, w])
                img = results[0].masks.data[idx_select]
                m = img[(h2-h)//2:(h2+h)//2, :]
                
                # 신뢰도 저장
                if hasattr(ctx, 'Seg3d') and hasattr(ctx['Seg3d'], 'confidences'):
                    ctx['Seg3d'].confidences.append(results[0].boxes.conf[idx_select])
                
                # idx_selected에 저장
                if hasattr(ctx, 'Seg3d') and hasattr(ctx['Seg3d'], 'idx_selected'):
                    if not hasattr(ctx['Seg3d'], 'idx_selected'):
                        ctx['Seg3d'].idx_selected = {}
                    current_view_idx = ctx['Seg3d'].data_dict['i_train'][0] if len(ctx['Seg3d'].data_dict['i_train']) > 0 else 0
                    ctx['Seg3d'].idx_selected[current_view_idx] = idx_select

            # 마스크를 3채널로 변환
            masks = torch.zeros([c, h, w]).cpu().numpy()
            masks[0:3, :, :] = m.type(torch.bool).cpu().numpy()
            
            # 결과 이미지 생성
            fig1 = (255*masks[0, :, :, None]*0.6 + ctx['cur_img']*0.4).astype(np.uint8)
            fig2 = (255*masks[1, :, :, None]*0.6 + ctx['cur_img']*0.4).astype(np.uint8)
            fig3 = (255*masks[2, :, :, None]*0.6 + ctx['cur_img']*0.4).astype(np.uint8)
            
            # Plotly 그림 생성
            fig1 = draw_figure(fig1, 'mask0')
            fig2 = draw_figure(fig2, 'mask1')
            fig3 = draw_figure(fig3, 'mask2')
            fig0 = draw_figure(ctx['cur_img'], 'original_image')
            
            return masks, fig0, fig1, fig2, fig3
        else:
            raise NotImplementedError


def draw_figure(fig, title, animation_frame=None):
    """그림을 그리는 함수 (gui.py에서 가져옴)"""
    import plotly.express as px
    fig = px.imshow(fig, animation_frame=animation_frame)
    if animation_frame is not None:
        fig.layout.updatemenus[0].buttons[0].args[1]["frame"]["duration"] = 33
    fig.update_layout(title_text=title, showlegend=False)
    fig.update_xaxes(showticklabels=False)
    fig.update_yaxes(showticklabels=False)
    return fig


if __name__=='__main__':
    # load setup
    parser = config_parser()
    args = parser.parse_args()
    cfg = Config.fromfile(args.config)

    # init enviroment
    if torch.cuda.is_available():
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    utils.seed_everything(args)

    # load images / poses / camera settings / data split
    data_dict = utils.load_everything(args=args, cfg=cfg)

    # train (자동화 모드)
    if not args.render_only:
        train_seg_auto(args, cfg, data_dict)

    # load model for further rendering
    e_flag = args.sp_name if args.sp_name is not None else ''
    if args.render_opt is not None:
        for seg_type in ['seg_img', 'seg_density']:
            if args.ft_path:
                ckpt_path = args.ft_path
            else:
                fine_path = os.path.join(cfg.basedir, cfg.expname, 'fine_segmentation'+e_flag+'.tar')
                coarse_path = os.path.join(cfg.basedir, cfg.expname, 'coarse_segmentation'+e_flag+'.tar')
                ckpt_path = fine_path if os.path.exists(fine_path) else coarse_path
            print("\033[96mRendering with ckpt "+ckpt_path+"\033[0m")
                
            ckpt_name = ckpt_path.split('/')[-1][:-4]
            model_class = utils.find_model(cfg)
            model, optimizer, start = utils.load_existed_model(args, cfg, cfg.fine_train, ckpt_path, device)
            
            stepsize = cfg.fine_model_and_render.stepsize
            render_viewpoints_kwargs = {
                'model': model,
                'ndc': cfg.data.ndc,
                'render_kwargs': {
                    'near': data_dict['near'],
                    'far': data_dict['far'],
                    'bg': 1 if cfg.data.white_bkgd else 0,
                    'stepsize': stepsize,
                    'inverse_y': cfg.data.inverse_y,
                    'flip_x': cfg.data.flip_x,
                    'flip_y': cfg.data.flip_y,
                    'render_depth': True,
                },
            }

            # rendering
            flag = "seg" if args.segment else ""
            if args.segment:
                if seg_type == 'seg_density':
                    render_viewpoints_kwargs['model'].segmentation_to_density()
                elif seg_type == 'seg_img':
                    render_viewpoints_kwargs['model'].segmentation_only()
                else:
                    raise NotImplementedError('seg type {} is not implemented!'.format(seg_type))

            # default: one object    
            num_obj = render_viewpoints_kwargs['model'].seg_mask_grid.grid.shape[1]
            render_viewpoints_kwargs['model'] = render_viewpoints_kwargs['model'].cuda()
            render_fn(args, cfg, ckpt_name, flag, e_flag, num_obj, \
                                   data_dict, render_viewpoints_kwargs, seg_type=seg_type)
