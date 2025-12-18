import os
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


def create_lens_image(radius=100, lens_color=(200, 200, 200), defect_type=None):
    """
    가상의 렌즈 이미지를 생성합니다.
    """
    img_size = 400
    img = np.zeros((img_size, img_size, 3), dtype=np.uint8)
    center = (img_size // 2, img_size // 2)

    # 렌즈 그리기 (동심원 패턴)
    # Zone C (Center)
    cv2.circle(img, center, int(radius * 0.4), (lens_color[0] - 100, lens_color[1] - 100, lens_color[2] - 100), -1)
    # Zone B (Middle)
    cv2.circle(img, center, int(radius * 0.7), (lens_color[0] + 50, lens_color[1] + 50, lens_color[2] + 50), 20)
    # Zone A (Edge)
    cv2.circle(img, center, int(radius * 0.95), (lens_color[0] - 50, lens_color[1] - 50, lens_color[2] - 50), 10)

    # 불량 주입
    if defect_type == "scratch":
        cv2.line(img, (center[0] - 20, center[1]), (center[0] + 20, center[1]), (255, 255, 255), 2)
    elif defect_type == "dot_missing":
        cv2.circle(img, (center[0] + 50, center[1]), 5, (0, 0, 0), -1)
    elif defect_type == "color_mismatch":
        # 색상을 약간 변경 (전체적으로)
        img = cv2.add(img, np.array([10, 10, 10], dtype=np.uint8))

    return img


def generate_data(sku, num_ok, num_ng, lens_color, metadata_list):
    output_dir = Path("data/raw_images")
    output_dir.mkdir(parents=True, exist_ok=True)

    # 양품 생성
    for i in range(num_ok):
        filename = f"{sku}_OK_{i+1:03d}.jpg"
        img = create_lens_image(lens_color=lens_color)
        cv2.imwrite(str(output_dir / filename), img)
        metadata_list.append({"file_name": filename, "sku": sku, "judgment": "OK", "description": "Normal sample"})

    # 불량 생성
    defects = ["scratch", "dot_missing", "color_mismatch", "scratch", "color_mismatch"]  # 5가지 불량 유형
    for i in range(num_ng):
        defect = defects[i % len(defects)]  # 불량 유형 순환
        filename = f"{sku}_NG_{i+1:03d}.jpg"
        img = create_lens_image(lens_color=lens_color, defect_type=defect)
        cv2.imwrite(str(output_dir / filename), img)
        metadata_list.append({"file_name": filename, "sku": sku, "judgment": "NG", "description": defect})

    print(f"Generated {num_ok + num_ng} images for SKU: {sku} in {output_dir}")


if __name__ == "__main__":
    all_metadata = []

    # SKU001 (기존 색상, 녹색 계열) - 10장
    generate_data("SKU001", 5, 5, (150, 150, 150), all_metadata)

    # SKU002 (파란색) - 20장 (OK 10장, NG 10장)
    generate_data("SKU002", 10, 10, (200, 100, 50), all_metadata)  # BGR

    # SKU003 (갈색) - 20장 (OK 10장, NG 10장)
    generate_data("SKU003", 10, 10, (70, 130, 200), all_metadata)  # BGR

    # 메타데이터 저장
    df = pd.DataFrame(all_metadata)
    df.to_csv(Path("data/raw_images") / "metadata.csv", index=False)
    print(f"Final metadata.csv generated with {len(all_metadata)} entries.")
