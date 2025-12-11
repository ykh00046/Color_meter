import cv2
import numpy as np
import pandas as pd
from pathlib import Path
import os

def create_lens_image(radius=100, color=(200, 200, 200), defect_type=None):
    """
    가상의 렌즈 이미지를 생성합니다.
    """
    img_size = 400
    img = np.zeros((img_size, img_size, 3), dtype=np.uint8)
    center = (img_size // 2, img_size // 2)
    
    # 렌즈 그리기 (동심원 패턴)
    # Zone C (Center)
    cv2.circle(img, center, int(radius * 0.4), (100, 50, 50), -1)
    # Zone B (Middle)
    cv2.circle(img, center, int(radius * 0.7), (50, 200, 200), 20)
    # Zone A (Edge)
    cv2.circle(img, center, int(radius * 0.95), (200, 100, 50), 10)

    # 불량 주입
    if defect_type == "scratch":
        cv2.line(img, (center[0]-20, center[1]), (center[0]+20, center[1]), (255, 255, 255), 2)
    elif defect_type == "dot_missing":
        cv2.circle(img, (center[0]+50, center[1]), 5, (0, 0, 0), -1)
    elif defect_type == "color_mismatch":
        # 색상을 약간 변경 (전체적으로)
        img = cv2.add(img, np.array([10, 10, 10], dtype=np.uint8))

    return img

def generate_data():
    output_dir = Path("data/raw_images")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    metadata = []
    
    # 양품 5장 생성
    for i in range(5):
        filename = f"OK_{i+1:03d}.jpg"
        img = create_lens_image()
        cv2.imwrite(str(output_dir / filename), img)
        metadata.append({"file_name": filename, "sku": "SKU001", "judgment": "OK", "description": "Normal sample"})

    # 불량 5장 생성
    defects = ["scratch", "dot_missing", "color_mismatch", "scratch", "color_mismatch"]
    for i, defect in enumerate(defects):
        filename = f"NG_{i+1:03d}.jpg"
        img = create_lens_image(defect_type=defect)
        cv2.imwrite(str(output_dir / filename), img)
        metadata.append({"file_name": filename, "sku": "SKU001", "judgment": "NG", "description": defect})

    # 메타데이터 저장
    df = pd.DataFrame(metadata)
    df.to_csv(output_dir / "metadata.csv", index=False)
    print(f"Generated 10 images and metadata.csv in {output_dir}")

if __name__ == "__main__":
    generate_data()
