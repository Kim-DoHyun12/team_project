# yolo11x를 이용한 자동 라벨링 코드
from ultralytics import YOLO
import cv2
import os

# 1. YOLO 모델 로드
model = YOLO("yolo11x.pt")  # COCO 기반 사전학습 모델

# 2. 이미지 폴더와 결과 저장 폴더 설정
image_folder = "team_project/captured_images"
output_image_folder = "team_project/results_images"
output_folder = "team_project/results_labels"

# 3. 결과 저장 폴더 생성
os.makedirs(output_folder, exist_ok=True)
os.makedirs(output_image_folder, exist_ok=True)

# 4. 클래스 ID 매핑 (COCO → 커스텀)
class_mapping = {
    0: 0,  # person → 0
    2: 1,  # car → 1
    3: 1,  # motorcycle → 1
    5: 1,  # bus → 1
    7: 1   # truck → 1
}

# IoU 계산 함수
def compute_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    if inter_area == 0:
        return 0.0

    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - inter_area
    return inter_area / union_area

# 5. 이미지 파일 목록 가져오기
image_files = [f for f in os.listdir(image_folder) if f.endswith(('.jpg', '.jpeg', '.png'))]

# 6. 각 이미지에 대해 라벨링 수행
for image_file in image_files:
    image_path = os.path.join(image_folder, image_file)

    # 이미지 로드 및 크기 확인
    image = cv2.imread(image_path)
    img_height, img_width = image.shape[:2]

    # 7. 모델 예측 (사람, 차, 오토바이 클래스만)
    results = model(image_path, classes=[0, 2, 3, 5, 7])

    boxes = results[0].boxes.xyxy.cpu().numpy()
    labels = results[0].boxes.cls.cpu().numpy()
    confidences = results[0].boxes.conf.cpu().numpy()

    # 사람과 차량 계열 박스 분리
    person_boxes = []
    vehicle_boxes = []
    all_boxes = []

    for box, label, confidence in zip(boxes, labels, confidences):
        class_id = int(label)
        mapped_id = class_mapping.get(class_id, None)
        if mapped_id is None:
            continue

        all_boxes.append((box, mapped_id))

        if class_id == 0:  # person
            person_boxes.append((box, confidence))
        elif class_id in [2, 3, 5, 7]:  # vehicle 계열
            vehicle_boxes.append(box)

    # 겹친 사람 박스 제거
    filtered_boxes = []
    for box, mapped_id in all_boxes:
        if mapped_id == 0:  # person
            overlapped = False
            for v_box in vehicle_boxes:
                iou = compute_iou(box, v_box)
                if iou > 0.3:  # IoU가 0.3 이상이면 겹침으로 판단
                    overlapped = True
                    break
            if overlapped:
                continue  # 제거
        filtered_boxes.append((box, mapped_id))

    # 9. 라벨 파일 생성
    label_file_path = os.path.join(output_folder, f"{os.path.splitext(image_file)[0]}.txt")
    with open(label_file_path, "w") as label_file:
        for box, mapped_class_id in filtered_boxes:
            x_center = (box[0] + box[2]) / 2 / img_width
            y_center = (box[1] + box[3]) / 2 / img_height
            width = (box[2] - box[0]) / img_width
            height = (box[3] - box[1]) / img_height

            label_file.write(f"{mapped_class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

    # 10. 시각화 이미지 저장
    image_with_labels = results[0].plot()
    result_image_path = os.path.join(output_image_folder, f"{os.path.splitext(image_file)[0]}.jpg")
    cv2.imwrite(result_image_path, image_with_labels)

    print(f"라벨링 완료: {image_file}")
    print(f"결과 이미지 저장 경로: {result_image_path}")

# 12. 완료 메시지
print("모든 이미지에 대한 자동 라벨링이 완료되었습니다.")
