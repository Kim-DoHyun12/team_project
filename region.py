# save this as app.py
from flask import Flask, Response
from ultralytics import solutions , YOLO
import cv2

app = Flask(__name__)

#모델 로드
model = YOLO("best2.pt")
print(model.names)
# 레기온 좌표
def generate_rgion():
    cap = cv2.VideoCapture("https://strm2.spatic.go.kr/live/174.stream/playlist.m3u8")
    region_points = {
        "region-01" : [(644, 375), (999, 409), (989, 451), (601, 413)], # 횡단보도 1 [(550, 394), (258, 422), (580, 452), (267, 479)]
        "region-02" : [(405, 427), (260, 542), (325, 624), (520, 434)], # 횡단보도 2
        "region-03" : [(1051, 553), (1207, 604), (1153, 718), (1011, 715)], # 횡단보도 3
        "region-04" : [(799, 321), (750, 379), (811, 393), (841, 341)], # 정지선 1.
        "region-05" : [(1042, 143), (1021, 259), (1055, 230), (1063, 179)]  # 정지선 2.
    }
    # 구역 설정
    region = solutions.RegionCounter(
        show = True,
        region = region_points,
        # region_model = model
        region_model = "best2.pt"
    )
    while True:
        success, frame = cap.read()
        if not success:
            print("frame check")
            break
        # 객체 탐지
        
        results = region(frame)
        # 탐지 표시
        # print(results)           # 객체 전체 출력
        # 또는 더 자세히 보고 싶으면
        # print(dir(results))     # 속성 확인
        
        annotated_frame = results.plot_im
        #객체 수 추출
        # detected_object_count = len(results[0].boxes)
        # detected_object_count = 1
        detected_object_count = sum(results.region_counts.values())
        status = f"COUNT : {detected_object_count}"
        color = (0, 255, 0)  # 기본 초록색
        
        # if detected_object_count <= 1:
        #     status += "=> Warning"
        #     color = (0,0,255)
            
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
        
        #reframe = cv2.resize(frame,(640, 480))
        # 프레임 인코딩
        _, buffer = cv2.imencode('.jpg', annotated_frame)
        #인코딩을 바이트
        frame_bytes = buffer.tobytes()     
            
        yield(b'--frame\r\n' b'Content-Type : image/jpeg\r\n\r\n' + frame_bytes + b'\r\n' )
    cap.release()
        

@app.route('/')    
def Region_pro():
    return Response(generate_rgion(), mimetype="multipart/x-mixed-replace; boundary=frame")

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)
# app 실행시 .py 실행 하는법
# if status <= 1:     # 1인 이하
            #return render_template('blue.html')