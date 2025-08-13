import os
import cv2
import numpy as np

# --- Configuration ---
IMG_DIR = 'data/train/'
PROCESSED_DIR = 'data/processed/'
IMG_SIZE = (128, 128) # Resize images to 128x128 pixels

# Create processed data directory if it doesn't exist
os.makedirs(PROCESSED_DIR, exist_ok=True)

# --- Preprocessing Loop ---
image_files = [f for f in os.listdir(IMG_DIR) if f.endswith(('.png', '.jpg', '.jpeg', '.bmp'))]

for filename in image_files:
    # Load image
    img_path = os.path.join(IMG_DIR, filename)
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE) # Load as grayscale

    # Resize image
    img_resized = cv2.resize(img, IMG_SIZE)

    # Save processed image
    save_path = os.path.join(PROCESSED_DIR, filename)
    cv2.imwrite(save_path, img_resized)

print(f"Processed {len(image_files)} images and saved them to {PROCESSED_DIR}")

# --- Load processed images for training ---
processed_images = []
processed_files = [f for f in os.listdir(PROCESSED_DIR) if f.endswith(('.png', '.jpg', '.jpeg' , '.bmp'))]

for filename in processed_files:
    img_path = os.path.join(PROCESSED_DIR, filename)
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    processed_images.append(img)

# Convert to numpy array and normalize
train_data = np.array(processed_images).astype('float32') / 255.0
# Add a channel dimension for the Conv2D layers
train_data = np.expand_dims(train_data, axis=-1)

# Save the numpy array for easy loading later
np.save('train_data.npy', train_data)
print(f"Training data shape: {train_data.shape}")