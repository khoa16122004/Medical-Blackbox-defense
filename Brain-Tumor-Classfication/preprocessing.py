import torch
import os
from config import *
from PIL import Image
import cv2 as cv
import numpy as np
import os
import cv2 as cv
import torch
from torchvision import transforms
from torchvision.utils import save_image
from tqdm import tqdm

def crop_image(img_path):
    img = cv.imread(img_path, cv.IMREAD_GRAYSCALE)
    original_img = img.copy()
    
    # finding contour and extremepoint
    blurred_img = cv.GaussianBlur(img, (5, 5), 0)
    _, thresh = cv.threshold(blurred_img, 45, 255, cv.THRESH_BINARY)
    thresh = cv.erode(thresh, None, iterations=2)
    thresh = cv.dilate(thresh, None, iterations=2)
    contours, _ = cv.findContours(thresh.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    largest_contour = max(contours, key=cv.contourArea)

    extLeft = tuple(largest_contour[largest_contour[:, :, 0].argmin()][0])
    extRight = tuple(largest_contour[largest_contour[:, :, 0].argmax()][0])
    extTop = tuple(largest_contour[largest_contour[:, :, 1].argmin()][0])
    extBot = tuple(largest_contour[largest_contour[:, :, 1].argmax()][0])

    # crop image
    cropped_img = original_img[extTop[1]:extBot[1], extLeft[0]:extRight[0]]

    return cropped_img

def create_new_dataset():        
    if not os.path.exists(TRAINING_NEW_DIR):
        os.mkdir(TRAINING_NEW_DIR)
        
    if not os.path.exists(TESTING_NEW_DIR):
        os.mkdir(TESTING_NEW_DIR)
    
    # create new dataset    
    for file_name in os.listdir(TRAINING_ORIGINAL_DIR):
        file_path = os.path.join(TRAINING_ORIGINAL_DIR, file_name)        
        crop_img = crop_image(file_path)
        
        output_path = file_path.replace('original', "new")
        cv.imwrite(output_path, crop_img)
        
    for file_name in os.listdir(TESTING_ORIGINAL_DIR):
        file_path = os.path.join(TESTING_ORIGINAL_DIR, file_name)        
        crop_img = crop_image(file_path)
        
        output_path = file_path.replace('original', "new")
        cv.imwrite(output_path, crop_img)

def argument_data():
    
    
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),             # Lật ngang ngẫu nhiên với xác suất 0.5
        transforms.RandomVerticalFlip(p=0.5),               # Lật dọc ngẫu nhiên với xác suất 0.5
        transforms.RandomRotation(degrees=45),              # Xoay ngẫu nhiên trong khoảng +/- 45 độ
        transforms.ColorJitter(brightness=0.5,              # Thay đổi độ sáng ngẫu nhiên
                               contrast=0.5,                # Thay đổi độ tương phản ngẫu nhiên
                               saturation=0.5,              # Thay đổi độ bão hòa ngẫu nhiên
                               hue=0.1)                     # Thay đổi tông màu ngẫu nhiên
    ])

    with open(ARGUMENT_PATH, "w+") as f_out, open(TRAINING_ANNOTATION_PATH, "r") as f:
        lines = f.readlines()
        lines = [line.strip() for line in lines]
        for line in tqdm(lines):
            file_name, label = line.split(",")
            base_name = file_name.split(".")[0]
            
            file_path = os.path.join(TRAINING_NEW_DIR, file_name)
            img = Image.open(file_path)
            img.save(os.path.join(ARGUMENT_DIR, f"{base_name}.jpg"))
            f_out.write(f"{file_name}, {label}\n")
            
            augmented_images = [transform(img) for _ in range(10)]  
            for i, img_transform in enumerate(augmented_images):
                output_path = os.path.join(ARGUMENT_DIR, f"{base_name}_{i}.jpg")
                f_out.write(f"{base_name}_{i}.jpg, {label}\n")
                img_transform.save(output_path)
    
    # for file_name in os.listdir(TRAINING_NEW_DIR):
    #     file_path = os.path.join(TRAINING_NEW_DIR, file_name)        
argument_data() 
# create_new_dataset()
    


    