import argparse
import cv2
from ultralytics import YOLO


def visualize(image_path: str, model_name: str = 'yolov8x-seg.pt'):
    """
    주어진 이미지 경로에서 YOLOv8-seg 모델로 추론을 수행하고 결과를 화면에 표시합니다.
    Qt 스레딩 이슈를 피하기 위해 startWindowThread 및 namedWindow를 사용합니다.
    :param image_path: 검출할 이미지 파일 경로
    :param model_name: 사전학습된 YOLOv8-seg 모델 이름 (기본값: 'yolov8x-seg.pt')
    """
    # 모델 로드
    model = YOLO(model_name)

    # 추론 수행
    results = model(image_path)[0]

    # 결과 시각화 (bbox + mask)
    annotated_frame = results.plot()

    window_name = 'YOLOv8-seg 결과'
    # 윈도우 스레드 시작 (Qt 이슈 방지)
    cv2.startWindowThread()
    # 윈도우 생성 (크기 조절 가능)
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    cv2.imshow(window_name, annotated_frame)
    # Qt 이벤트 처리
    cv2.waitKey(1)
    # 사용자 키 입력 대기
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(
        description='YOLOv8-seg로 이미지 객체 검출 및 시각화'
    )
    parser.add_argument('image', help='검출할 이미지 파일 경로')
    parser.add_argument(
        '--model', default='yolov8x-seg.pt',
        help='사용할 사전학습 YOLO 모델 파일 (기본: yolov8x-seg.pt)'
    )
    args = parser.parse_args()

    visualize(args.image, args.model)


if __name__ == '__main__':
    main()
