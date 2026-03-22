import cv2
import os
import numpy as np
from PIL import Image
import albumentations as A

# ==== CONFIG ====
INPUT_IMAGE_PATH = "input.png"
OUTPUT_FOLDER = "augmented_images"
OUTPUT_COUNT = 200
RESIZE_SHAPE = (256, 256)  # (width, height)
# ===============

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Load and resize
image = cv2.imread(INPUT_IMAGE_PATH)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image = cv2.resize(image, RESIZE_SHAPE)

# Define augmentation pipeline
transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.2),
    A.RandomRotate90(p=0.5),
    A.Rotate(limit=25, p=0.5),
    A.RandomBrightnessContrast(p=0.5),
    A.GaussianBlur(blur_limit=(3, 7), p=0.3),
    A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
    A.RandomScale(scale_limit=0.2, p=0.4),
])

# Generate and save
for i in range(OUTPUT_COUNT):
    augmented = transform(image=image)
    aug_image = augmented["image"]
    img_pil = Image.fromarray(aug_image)
    img_pil.save(os.path.join(OUTPUT_FOLDER, f"aug_{i+1:03d}.png"))

print(f"✅ {OUTPUT_COUNT} augmented images saved in '{OUTPUT_FOLDER}'")
