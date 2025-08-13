import os
import cv2

# The root directory of your dataset
ROOT_DIR = 'classifier_data/train/other_machinery'

print("Starting image conversion...")

# Walk through all directories and subdirectories
for dirpath, _, filenames in os.walk(ROOT_DIR):
    for filename in filenames:
        if filename.lower().endswith('.bmp'):
            # Construct full file path
            bmp_path = os.path.join(dirpath, filename)
            
            # Create new filename with .png extension
            png_filename = os.path.splitext(filename)[0] + '.png'
            png_path = os.path.join(dirpath, png_filename)
            
            try:
                # Read the BMP image
                img = cv2.imread(bmp_path)
                if img is not None:
                    # Save the image as PNG
                    cv2.imwrite(png_path, img)
                    print(f"Converted {bmp_path} to {png_path}")
                    # Optional: remove the old bmp file
                    os.remove(bmp_path)
                else:
                    print(f"Could not read {bmp_path}")
            except Exception as e:
                print(f"Error converting {bmp_path}: {e}")

print("Conversion complete.")