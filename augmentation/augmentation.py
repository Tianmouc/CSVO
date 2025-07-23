import cv2
import numpy as np
import random

def random_blur(image):
    # 随机选择模糊核大小
    kernel_size = random.choice([3, 5, 7])
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)

def random_hdr(image):
    # 模拟HDR效果
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    lab = cv2.merge((l, a, b))
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

def random_exposure(image):
    # 随机选择过曝光或欠曝光
    exposure_type = random.choice(['over', 'under'])
    gamma = random.uniform(0.5, 4) if exposure_type == 'over' else random.uniform(0.4, 0.6)
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)

def random_augmentation(image):
    # 随机选择一种数据增强方法
    augmentation = random.choice([random_blur, random_hdr, random_exposure])
    for augmentation in [random_blur, random_hdr, random_exposure]:
        image = augmentation(image)
    return image

test_split = [
    "abandonedfactory/Easy/P011",
    "abandonedfactory/Hard/P011",
    "abandonedfactory_night/Easy/P013",
    "abandonedfactory_night/Hard/P014",
    "amusement/Easy/P008",
    "amusement/Hard/P007",
    "carwelding/Easy/P007",
    "endofworld/Easy/P009",
    "gascola/Easy/P008",
    "gascola/Hard/P009",
    "hospital/Easy/P036",
    "hospital/Hard/P049",
    "japanesealley/Easy/P007",
    "japanesealley/Hard/P005",
    "neighborhood/Easy/P021",
    "neighborhood/Hard/P017",
    "ocean/Easy/P013",
    "ocean/Hard/P009",
    "office2/Easy/P011",
    "office2/Hard/P010",
    "office/Hard/P007",
    "oldtown/Easy/P007",
    "oldtown/Hard/P008",
    "seasidetown/Easy/P009",
    "seasonsforest/Easy/P011",
    "seasonsforest/Hard/P006",
    "seasonsforest_winter/Easy/P009",
    "seasonsforest_winter/Hard/P018",
    "soulcity/Easy/P012",
    "soulcity/Hard/P009",
    "westerndesert/Easy/P013",
    "westerndesert/Hard/P007",
]

def main(image_path = '/data/zzx/DPVO_E2E/datasets/TartanAirNew/abandonedfactory_night/Easy/P002/image_left/000000_left.png',output_path = 'augmented_image.jpg'):
    # 读取图片
    
    image = cv2.imread(image_path)

    if image is None:
        print("Image loading failed, please check the path.")
        return

    # 随机数据增强
    augmented_image = random_augmentation(image)

    # 保存增强后的图片
    
    cv2.imwrite(output_path, augmented_image)
    # print(f"增强后的图片已保存到: {output_path}")
import os
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script to process TartanAir dataset path.")
    parser.add_argument("dataset_path", type=str, help="Path to the TartanAir dataset")
    args = parser.parse_args()

    dataset_path = args.dataset_path
    for test in test_split:
        image_dir = f"{dataset_path}/{test}/image_left"
        for image in os.listdir(image_dir):
            image_path = os.path.join(image_dir, image)
            output_path = f"{dataset_path}/{test}/augmented_image/{image}"
            if os.path.exists(output_path):
                continue
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            main(image_path, output_path)
        print(test, 'done')