from ultralytics import solutions
import cv2

cap = cv2.VideoCapture("road-person.mp4")

region_points = {
    "region-01": [(290,471),(227,511),(1068,719),(1039,616)],
    #"region-02": [(150,60),(150,300),(300,300),(300,60)],
    #"region-03": [(300,60),(300,300),(500,300),(500,60)]
}

region = solutions.RegionCounter(
    model="team_project/best100.pt",
    show=True,
    region=region_points
)

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("Frame Check")
        break

    frame_resized = cv2.resize(frame, (1280,720))
    #print(im0)
    #print(dir(im0))
    
    region1 = region.region_counts.get(("region-01"),0)
    region2 = region.region_counts.get(("region-02"),0)
    region3 = region.region_counts.get(("region-03"),0)
    print(f"region1:{region1}")
    #print(f"region2:{region2}")
    #print(f"region3:{region3}")
    
    cv2.putText(
            frame_resized,
            f"region1:{region1}", #텍스트, region2:{region2}, region3:{region3}
            (100,30), #위치
            cv2.FONT_HERSHEY_SIMPLEX,
            1, #굵기
            (0,0,255),
            2,
            cv2.LINE_AA
        )
    cv2.imshow(f"Ultralytics Solutions", frame_resized)
    im0 = region(frame_resized)
    
    key = cv2.waitKey(1)
    if key & (0xFF == ord('q') or key==27):
        break

cap.release()
cv2.destroyAllWindows()

#1.터미널 구역 탐지객체 수, 2.영상 구역 탐지객체 수