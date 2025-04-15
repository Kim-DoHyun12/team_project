# cctv 화면 캡쳐 코드
import cv2
import os

video_url = "https://cctv.fitic.go.kr/cctv/L306.stream/playlist.m3u8"
output_dir = "team_project\cctv_frames"
os.makedirs(output_dir, exist_ok=True)

# 기존 이미지 개수 확인해서 번호 이어 붙이기
existing_images = [f for f in os.listdir(output_dir) if f.endswith(".jpg")]
existing_images.sort()
start_num = len(existing_images)

cap = cv2.VideoCapture(video_url)

frame_interval = 30  # 프레임 간격
saved_count = 0
frame_id = 0
max_frames = 10  # 저장할 새 프레임 수

while cap.isOpened() and saved_count < max_frames:
    ret, frame = cap.read()
    if not ret:
        break

    if frame_id % frame_interval == 0:
        filename = os.path.join(output_dir, f"frame_{start_num + saved_count + 1}.jpg")
        cv2.imwrite(filename, frame)
        saved_count += 1

    frame_id += 1

cap.release()
print(f"{saved_count}개의 프레임이 저장되었습니다.")
