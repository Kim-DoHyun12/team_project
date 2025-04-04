import cv2
import numpy as np

# 단계 1: 비디오 파일 열기
video = cv2.VideoCapture("cross.mp4")

# 단계 2: 차선 검출
def detect_lane(image):
    # 이미지 전처리

    # 효과적인 경계 검출을 위해 그레이 스케일로 변환
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 가우시안 블러를 적용하여 이미지를 부드럽게 만듦
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # 엣지 검출을 위해 캐니 알고리즘 적용
    edges = cv2.Canny(blurred, 50, 150)

    # 관심 영역 설정

    # 교통 영역 관심 영역의 꼭지점 좌표
    height, width = image.shape[:2]
    roi_vertices = [(0, height), (width // 2, height // 2), (width, height)]

    # 마스크 생성
    mask = np.zeros_like(edges)
    cv2.fillPoly(mask, np.array([roi_vertices], dtype=np.int32), 255)

    # 관심 영역 적용
    masked_edges = cv2.bitwise_and(edges, mask)

    # 차선 검출 알고리즘

    # 허프 변환을 이용한 직선 검출
    lines = cv2.HoughLinesP(masked_edges, rho=2, theta=np.pi/180, threshold=100, minLineLength=100, maxLineGap=50)

    # 검출된 차선 시각화
    lane_image = np.zeros_like(image)
    draw_lines(lane_image, lines)

    return lane_image

def draw_lines(image, lines, color=(0, 255, 0), thickness=5):
    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(image, (x1, y1), (x2, y2), color, thickness)

# 단계 3: 차량 및 보행자 감지
def detect_objects(image):
    # YOLO 모델 로드
    net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")

    # 클래스 이름 로드
    with open("coco.names", "r") as f:
        classes = [line.strip() for line in f.readlines()]

    # 이미지 전처리 및 객체 감지 수행
    blob = cv2.dnn.blobFromImage(image, 1/255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    layer_names = net.getLayerNames()
    print(dir(layer_names))
    output_layers = [layer_names[i- 1] for i in net.getUnconnectedOutLayers()]
    outputs = net.forward(output_layers)

    # 객체 감지 결과 시각화
    confidence_threshold = 0.5
    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > confidence_threshold:
                center_x = int(detection[0] * image.shape[1])
                center_y = int(detection[1] * image.shape[0])
                width = int(detection[2] * image.shape[1])
                height = int(detection[3] * image.shape[0])
                x = int(center_x - width / 2)
                y = int(center_y - height / 2)

                cv2.rectangle(image, (x, y), (x+width, y+height), (0, 255, 0), 2)
                cv2.putText(image, classes[class_id], (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    return image

# 단계 4: 결과 시각화
while True:
    ret, frame = video.read()
    if not ret:
        break

    lane_image = detect_lane(frame)
    objects_image = detect_objects(frame)

    result_image = cv2.addWeighted(lane_image, 0.8, objects_image, 0.2, 0)

    cv2.imshow("Traffic Analysis", result_image)

    if cv2.waitKey(1) == ord('q'):
        break

video.release()
cv2.destroyAllWindows()