# 이미지 랜덤 증강 코드
import os
import cv2
import imgaug.augmenters as iaa
import numpy as np

# 폴더 경로 설정
input_folder = "team_project/original_images"          # 원본 이미지 폴더
output_folder = "team_project/augmented_images"        # 증강 이미지 저장 폴더
os.makedirs(output_folder, exist_ok=True)

# 증강기 정의 (랜덤하게 여러 조합으로)
augmenters = iaa.Sequential([
    iaa.SomeOf((1, 4), [
        iaa.Affine(rotate=(-25, 25)),
        iaa.Fliplr(0.5),
        iaa.GaussianBlur((0, 1.5)),
        iaa.Multiply((0.7, 1.3)),  # 밝기
        iaa.LinearContrast((0.6, 1.4)),
        iaa.AdditiveGaussianNoise(scale=(0, 0.05*255)),
        iaa.ScaleX((0.8, 1.2)),
        iaa.ScaleY((0.8, 1.2)),
        iaa.Crop(percent=(0, 0.1))
    ])
])

# 이미지 파일 리스트
image_files = [f for f in os.listdir(input_folder) if f.endswith(('.jpg', '.png', '.jpeg'))]

# 증강
for idx, filename in enumerate(image_files):
    image_path = os.path.join(input_folder, filename)
    image = cv2.imread(image_path)

    n = 10
    for i in range(n):  # 이미지당 n장 생성
        augmented_image = augmenters(image=image)
        out_filename = f"{os.path.splitext(filename)[0]}_aug_{i+1}.jpg"
        out_path = os.path.join(output_folder, out_filename)
        cv2.imwrite(out_path, augmented_image)

    print(f"{filename} → {n}장 증강 완료 ✅")

print("\n🎉 모든 이미지 증강 완료! 총 생성된 이미지 수:", len(image_files)*n)
