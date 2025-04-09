# cmd 창 => conda activate py39
# pip install bing-image-downloader

import os
import glob
from bing_image_downloader import downloader

def download_images(keyword, limit):
    downloader.download(keyword, limit=limit, output_dir='images', adult_filter_off=True, force_replace=False)

    # 다운로드된 폴더 경로
    download_path = os.path.join("images", keyword)
    
    # .jpg가 아닌 파일 삭제
    for file in glob.glob(os.path.join(download_path, "*")):
        if not file.lower().endswith(".jpg"):
            os.remove(file)

keyword = input("이미지 검색어를 입력하세요: ")
limit = int(input("다운로드할 이미지 수를 입력하세요: "))
download_images(keyword, limit)