# save this as app.py
from flask import Flask, Response
from ultralytics import solutions , YOLO
import cv2
app = Flask(__name__)

#모델 로드
model = YOLO("yolo11n.pt")
# 레기온 좌표
def generate_rgion():
    cap = cv2.VideoCapture("https://strm4.spatic.go.kr/live/343.stream/playlist.m3u8")
    region_points = {
        "region-01" : [(392, 151), (160, 369), (189, 439), (441, 159)], # 횡단보도 1
        "region-02" : [(462, 152), (635, 250), (637, 213), (482, 132)], # 횡단보도 2
        "region-03" : [(146, 333), (205, 281), (210, 283), (149, 337)], # 정지선 1.
        "region-04" : [(571, 158), (490, 122), (485, 125), (569, 165)]  # 정지선 2.
    }
    # 구역 설정
    region = solutions.RegionCounter(
        show = True,
        region = region_points,
        region_model = model
    )
    while True:
        success, frame = cap.read()
        if not success:
            print("frame check")
            break
        # 객체 탐지
        results = model(frame, conf=0.6)
        results_r = region(frame)
        # 탐지 표시
        annotated_frame = results[0].plot()
        #annotated_frame_r = results_r.plot_im
        #객체 수 추출
        detected_object_count = len(results[0].boxes)
        #detected_object_count_r = results_r.total_tracks
        status = f"COUNT : {detected_object_count}"
        color = (0,0,0)
        if detected_object_count <= 1:
            status += "=> Warning"
            color = (0,0,255)
        cv2.putText(
            annotated_frame,
            f"{status}",
            (10,30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            color,
            2
            #cv2.LINE_AA 
        )
        region01 = region.region_counts.get(("region-01"))
        region02 = region.region_counts.get(("region-02"))
        region03 = region.region_counts.get(("region-03"))
        region04 = region.region_counts.get(("region-04"))
        print(f"region01 person : {region01}")
        print(f"region02 person : {region02}")
        print(f"region03 person : {region03}")
        print(f"region04 person : {region04}")
        #reframe = cv2.resize(frame,(640, 480))
        # 프레임 인코딩
        _, buffer = cv2.imencode('.jpg', annotated_frame)
        #인코딩을 바이트
        frame_bytes = buffer.tobytes()
            
        yield(b'--frame\r\n' b'Content_Type : image/jpeg\r\n\r\n' + frame_bytes + b'\r\n' )
    cap.release()
    #cv2.destroyWindow("Ultralytics Solutions")
    
        

@app.route('/')    
def Region_Pro():
    return Response(generate_rgion(), mimetype="multipart/x-mixed-replace; boundary=frame")

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)
    
    