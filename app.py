import cv2
import numpy as np
from ultralytics import YOLO
from collections import deque
from datetime import datetime
import os
import sys
import io

# YOLO 모델 로드 (Load the YOLO model)C:\Users\Administrator\Desktop\AI\wtdc\team_project\best3.pt
model = YOLO("team_project/best3.pt")
class_names = model.names  # YOLO 모델의 클래스 이름들 (class names of the YOLO model)

# 마우스 클릭 이벤트를 위한 전역 변수 (Global variables for mouse events)
drawing = False
points = []  # 저장된 포인트 (points for the region)
region_data = {}  # 정의된 영역을 저장하는 딕셔너리 (Dictionary to store defined regions)
region_counter = {"stop_line": 1, "illegal parking": 1, "safe_zone": 1}  # 영역 이름 카운터 (Counter for region names)
violated_regions = {}  # 각 영역에서 위반한 객체 ID 저장 (Store violated object IDs per region)
violation_counts = {}  # 위반 횟수 저장 (Store violation counts)

# 위반 이미지 저장 폴더 생성 (Create folder to save violation images)
os.makedirs("captures", exist_ok=True)

# 마우스 클릭 이벤트 콜백 (Mouse click callback function)
def mouse_callback(event, x, y, flags, param):
    global points
    if event == cv2.EVENT_LBUTTONDOWN:  # 좌클릭으로 점 추가 (Add point with left click)
        points.append((x, y))

# 영역을 선택하는 GUI 함수 (Function to select a region with GUI)
def select_region_from_frame(frame):
    global points, region_data, region_counter
    clone = frame.copy()
    cv2.namedWindow("Define Region")  # 새 윈도우 생성 (Create new window)
    cv2.setMouseCallback("Define Region", mouse_callback)  # 마우스 클릭 콜백 설정 (Set mouse click callback)

    while True:
        temp = clone.copy()

        # 이미 정의된 영역 표시 (Display previously defined regions)
        for name, info in region_data.items():
            color = (0, 0, 255) if info["type"] == "stop_line" else \
                    (0, 255, 255) if info["type"] == "illegal parking" else \
                    (0, 255, 0)
            cv2.polylines(temp, [np.array(info["points"])], True, color, 2)
            cv2.putText(temp, name, info["points"][0], cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)

        # 현재 그리고 있는 영역 표시 (Display the region being drawn)
        if len(points) > 1:
            cv2.polylines(temp, [np.array(points)], False, (255, 255, 255), 1)

        # 화면에 안내 메시지 표시 (Display guide message)
        cv2.putText(temp,
                    "Draw: L-click | Save stop_line: S | Save illegal parking: P | Save safe_zone: Z | Reset: R | Finish: Enter",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (80,127,255), 2)
        cv2.imshow("Define Region", temp)

        key = cv2.waitKey(1)  # 키 입력 대기 (Wait for key input)

        if key == 13 and region_data:  # Enter 키로 영역 선택 종료 (Finish region selection with Enter)
            break
        elif key == ord('r'):  # R 키로 초기화 (Reset with 'r')
            region_data.clear()
            region_counter = {"stop_line": 1, "illegal parking": 1, "safe_zone": 1}
            points.clear()
        elif key == ord('s') and len(points) >= 3:  # S 키로 stop_line 저장 (Save stop_line with 's')
            name = f"stop_line_{region_counter['stop_line']}"
            region_data[name] = {"points": points.copy(), "type": "stop_line"}
            region_counter["stop_line"] += 1
            points.clear()
        elif key == ord('p') and len(points) >= 3:  # P 키로 parking 저장 (Save parking with 'p')
            name = f"illegal parking{region_counter['illegal parking']}"
            region_data[name] = {"points": points.copy(), "type": "illegal parking"}
            region_counter["illegal parking"] += 1
            points.clear()
        elif key == ord('z') and len(points) >= 3:  # Z 키로 safe_zone 저장 (Save safe_zone with 'z')
            name = f"safe_zone_{region_counter['safe_zone']}"
            region_data[name] = {"points": points.copy(), "type": "safe_zone"}
            region_counter["safe_zone"] += 1
            points.clear()

    cv2.destroyAllWindows()  # 윈도우 종료 (Close the window)

# 특정 점이 영역 안에 있는지 확인 (Check if a point is inside a region)
def inside_region(x, y, polygon):
    return cv2.pointPolygonTest(np.array(polygon), (x, y), False) >= 0

# 위반 이미지 저장 (Save violation image)
def save_violation_image(frame, region_name, track_id):
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')  # 현재 시간으로 파일명 생성 (Generate filename with timestamp)
    filename = f"captures/{region_name}_id{track_id}_{timestamp}.jpg"
    cv2.imwrite(filename, frame)  # 이미지 저장 (Save image)

# 전체 영상 분석 및 위반 감지 (Analyze entire video and detect violations)
def generate_all():
    from tkinter import filedialog
    fn = filedialog.askopenfilename(filetypes=(("MP4","*.mp4"),))
    print(fn)
    
    global violation_counts, violated_regions
    #1.정지선 : "C:/Users/Administrator/Desktop/AI/wtdc/team_project/video_file/Jeong Yeomcheon Bridge 3.mp4"
    #2.안전지대 : "C:/Users/Administrator/Desktop/AI/wtdc/team_project/video_file/An Bokgaecheon intersection.mp4"
    #3.불법주정차 : "C:/Users/Administrator/Desktop/AI/wtdc/team_project/video_file/heatmap.mp4"
    cap = cv2.VideoCapture(fn)  # 영상 열기 (Open video)

    # 첫 유효 프레임에서 영역 설정 (Set regions on the first valid frame)
    for _ in range(10):
        ret, frame = cap.read()
        if ret:
            frame = cv2.resize(frame, (1280, 720))  # 프레임 크기 조정 (Resize frame)
            select_region_from_frame(frame)  # 영역 선택 함수 호출 (Call the region selection function)
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # 첫 프레임으로 돌아가기 (Go back to first frame)
            break
    else:
        print("유효한 프레임 없음")  # 유효한 프레임을 찾을 수 없을 경우 (If no valid frame is found)
        return

    fourcc = cv2.VideoWriter_fourcc(*'XVID')  # 비디오 포맷 설정 (Set video format)
    out = cv2.VideoWriter('team_yolo-parking2.avi', fourcc, 30.0, (1280, 720))  # 결과 영상 저장 (Save output video)

    violated_regions = {name: set() for name in region_data}  # 각 영역별 위반 객체 ID 초기화 (Initialize violated object IDs per region)
    violation_counts = {name: {} for name in region_data}  # 각 영역별 위반 횟수 초기화 (Initialize violation counts per region)
    person_buffer = deque(maxlen=15)  # 사람 감지 여부를 추적하는 버퍼 (Buffer to track person detection)

    while True:
        ret, frame = cap.read()
        if not ret:
            break  # 영상 끝 (End of video)

        frame = cv2.resize(frame, (1280, 720))  # 프레임 크기 조정 (Resize frame)
        annotated = frame.copy()  # 주석을 추가한 복사본 (Copy for annotations)

        region_polys = {name: np.array(info["points"], dtype=np.int32) for name, info in region_data.items()}  # 영역 다각형 저장 (Store region polygons)

        # 정의된 영역에 선 그리기 (Draw the defined regions)
        for name, poly in region_polys.items():
            rtype = region_data[name]['type']
            color = (0, 0, 255) if rtype == "stop_line" else (0, 255, 255) if rtype == "illegal parking" else (0, 255, 0)
            cv2.polylines(annotated, [poly], True, color, 1)
            cv2.putText(annotated, name, poly[0], cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)

        results = model.track(frame, persist=True, conf=0.4, imgsz=640, tracker="botsort.yaml", verbose=False)  # YOLO 추적 (YOLO tracking)

        person_present = False  # 사람 감지 여부 (Person detection flag)

        if results and hasattr(results[0], "boxes") and results[0].boxes.id is not None:  # YOLO 결과에서 객체 추출 (Extract objects from YOLO results)
            boxes = results[0].boxes
            for box, cls, tid in zip(boxes.xyxy, boxes.cls, boxes.id):  # 바운딩 박스 및 클래스 정보 (Bounding box and class information)
                x1, y1, x2, y2 = map(int, box)
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2  # 중앙 좌표 (Center coordinates)
                tid = int(tid)
                class_name = class_names[int(cls)]  # 클래스 이름 (Class name)

                if class_name == "person":  # 사람 객체일 경우 (If it's a person)
                    person_present = True

                if class_name not in ["person", "objects"]:  # 사람과 객체만 추적 (Track only persons and objects)
                    continue

                violated = False  # 이번 프레임에서 위반 여부 (Violation status for this frame)

                # 각 영역에서 위반 여부 체크 (Check for violations in each region)
                for region_name, poly in region_polys.items():
                    if inside_region(cx, cy, poly):  # 객체가 영역 안에 있는지 확인 (Check if object is inside the region)
                        rtype = region_data[region_name]["type"]

                        if rtype == "stop_line":  # stop_line 영역 위반 처리 (Handle stop_line violations)
                            if person_present and tid not in violated_regions[region_name]:
                                violated_regions[region_name].add(tid)
                                violation_counts[region_name][tid] = violation_counts[region_name].get(tid, 0) + 1
                                save_violation_image(annotated, region_name, tid)
                                violated = True

                        elif rtype == "illegal parking" and class_name == "objects":  # 주차 위반 처리 (Handle parking violations)
                            if "entry_time" not in violation_counts[region_name]:
                                violation_counts[region_name]["entry_time"] = {}
                            current_time = datetime.now()
                            if tid not in violation_counts[region_name]["entry_time"]:
                                violation_counts[region_name]["entry_time"][tid] = current_time
                            entry_time = violation_counts[region_name]["entry_time"][tid]
                            time_diff = (current_time - entry_time).total_seconds()
                            if time_diff >= 10 and tid not in violated_regions[region_name]:
                                violated_regions[region_name].add(tid)
                                violation_counts[region_name][tid] = violation_counts[region_name].get(tid, 0) + 1
                                save_violation_image(annotated, region_name, tid)
                                violated = True

                        elif rtype == "safe_zone" and class_name == "objects":  # 안전 구역 위반 처리 (Handle safe zone violations)
                            if tid not in violated_regions[region_name]:
                                violated_regions[region_name].add(tid)
                                violation_counts[region_name][tid] = violation_counts[region_name].get(tid, 0) + 1
                                save_violation_image(annotated, region_name, tid)
                                violated = True

                # 위반한 객체에 빨간 바운딩 박스 표시 (Draw red bounding box for violated objects)
                if any(tid in violated_regions[region] for region in region_data):
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.putText(annotated, f"VIOLATION {tid}", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                else:
                    # 일반 객체는 파란색/주황색 바운딩 박스 표시 (Normal objects with blue/orange bounding box)
                    color = (255, 128, 0) if class_name == "person" else (255, 255, 0)
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(annotated, f"{class_name} ID:{tid}", (x1, y1 - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)

        person_buffer.append(person_present)

        # 화면에 메시지 및 위반 ID 표시 (Display message and violation IDs on the screen)
        y_offset = 50
        if person_present:
            cv2.putText(annotated, "person detected", (10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
            y_offset += 30

        for region_name, id_set in violated_regions.items():
            if id_set:
                text = f"{region_name}: " + ", ".join(map(str, id_set))
                cv2.putText(annotated, text, (10, y_offset),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
                y_offset += 30

        cv2.imshow("Violation Detection", annotated)  # 위반 감지 결과 표시 (Show violation detection results)
        out.write(annotated)  # 결과를 비디오 파일로 저장 (Save results to video file)

        if cv2.waitKey(1) & 0xFF == ord('q'):  # q 키로 종료 (Exit on 'q')
            break

    cap.release()  # 비디오 캡처 객체 해제 (Release video capture object)
    out.release()  # 비디오 출력 객체 해제 (Release video output object)
    cv2.destroyAllWindows()  # 모든 윈도우 닫기 (Close all windows)

if __name__ == "__main__":
    generate_all()  # 전체 비디오 분석 및 위반 감지 시작 (Start video analysis and violation detection)