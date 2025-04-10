import cv2
from ultralytics import solutions
from ultralytics import YOLO
import threading



video_region = "parking.mp4"
video_heatmap = "parking.mp4"






# 레기온 좌표 설정
region_points = {
    "region-01  " : [(242, 76), (239, 174),(289, 175),(290, 76)]
}

# 히트 맵 객체(불법 주정차)
heatmap = solutions.Heatmap(
    model = "yolo11n.pt",
     show= True,
     colormap = cv2.COLORMAP_PLASMA,
     #classes = 0,
     conf = 0.6,
)

# 레기온 객체(횡단 보도 침범)
region = solutions.RegionCounter(
    model = "yolo11n.pt",
    region = region_points,
    show = True
) 

# 히트맵 함수
def run_heatmap(filename):
    cap = cv2.VideoCapture(filename)
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("비디오 프레임 확인")
            break
        # 영상 출력 사이즈
        frame_resized = cv2.resize(frame,(640,480))
        cv2.namedWindow("HeatMap Test", cv2.WINDOW_NORMAL)
        heat_frame = heatmap(frame_resized)
        print("HeatMap", heat_frame)
        cv2.waitKey(1)
        
    cap.release()
    cv2.destroyAllWindows()     




# 레기온 함수ㅡ
def run_region(filename):
    cap = cv2.VideoCapture(filename)
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
        region01 = region.region_counts.get(("region-01"),0)
        print(f"person_num :  {region01}" )
        
        reframe = cv2.resize(frame,(640, 480))
        cv2.namedWindow("Region_Test", cv2.WINDOW_NORMAL)
        
        cv2.putText( # 텍스트 추가
            reframe , # 어디 화면에 넣어줄지
            f"region01 person : {region01}", #(텍스트)
            (10, 30), # (표시 위치)
            cv2.FONT_HERSHEY_SIMPLEX, #(폰트)
            1, #(텍스트 굵기)
            (0, 255, 0), #(텍스트 색상)
            2, #(텍스트 두께)
            cv2.LINE_AA #(텍스트 경계 선 타입 (LINE_AA는 부드러운 경계)
            )    
        cv2.waitKey(1)
        im0 = region(reframe)
    cap.release()
    cv2.destroyAllWindows()


#print(f"succes",{run_heatmap})
# 멀티 스레드 실행
thread_region = threading.Thread(target=run_region, args=(video_region,), daemon=True)
thread_heatmap = threading.Thread(target=run_heatmap, args=(video_heatmap,), daemon=True)

#스레드 시작
thread_region.start()
thread_heatmap.start()

#스레드 종료
thread_region.join()
thread_heatmap.join()





