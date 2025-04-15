# 모델 트레이닝 코드
from ultralytics import YOLO

# 1. yolo 모델 로드
model = YOLO("yolo11n.pt")

# 2. 모델 훈련
model.train(
    epochs=10,
    data="team_project/coco8.yaml"
)

# 1. 데이터셋 구축
    # - 라벨링
    # - train / val : 7 / 3
    # - 모델 학습
    # - 모델 확인
