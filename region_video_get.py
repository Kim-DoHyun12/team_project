import cv2
import time

vod_path = "road-person.mp4"
cap = cv2.VideoCapture(vod_path)

points = []

def mouse_callbak(event, x, y, flags, param):
    if event==cv2.EVENT_LBUTTONDOWN:
        points.append((x,y))
        print(f"Clicked:{x},{y}")
        
        '''if len(points)==6:
            print(f"Points:{points}")
            cv2.rectangle()
            points.clear()'''
    
cv2.namedWindow("Video_Get_X_Y")
cv2.setMouseCallback("Video_Get_X_Y",mouse_callbak)

while True:
    success, frame = cap.read()
    if not success:
        print("Frame Check")
        time.sleep(10)
        break

    re_frame = cv2.resize(frame, (720, 480))
    #cv2.namedWindow("Video_Get_X_Y", cv2.WINDOW_NORMAL)
    cv2.imshow("Video_Get_X_Y", re_frame)
    
    key = cv2.waitKey(1)
    if key & (0xFF == ord('q') or key==27):
        break

cap.release()
cv2.destroyAllWindows()

#Clicked:87,65 / 542,78 / 160,330 / 515,356