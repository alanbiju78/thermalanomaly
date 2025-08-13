import os
import numpy as np
import matplotlib.pyplot as plt
import glob
import cv2

path = os.getcwd()


# FOR ADAS use the 16-bit images
# M3FD only provide us the 8-bit images

for data in ["ADAS" , "M3FD"]:
    original_sub_folder = "thermal_16_bit" if data == "ADAS" else "Ir"
    
    for mod_sub_folder in ["train", "valid"]:

        modifications_path = path + f"/{data}/Modified/{mod_sub_folder}/modifications/"
        modified_images_path = path + f"/{data}/Modified/{mod_sub_folder}/images/"
        original_images_path = path + f"/{data}/Original/{original_sub_folder}/"

        image_files = [file for ext in ["*.jpeg", "*.tiff", "*.png"] for file in glob.glob(original_images_path + ext)]
        numpy_files = glob.glob(modifications_path + "*.npy")

        # Process each image and corresponding label
        for image_file in image_files:
            
            for numpy_file in numpy_files:
                        
                if os.path.splitext(image_file)[0].split('/')[-1] == os.path.splitext(numpy_file)[0].split('/')[-1]:
                    # Load the image
                    image = cv2.imread(image_file, cv2.IMREAD_UNCHANGED)
                    
                    # If the image is 16-bit, convert to 8-bit using cv2.normalize
                    if image.dtype == np.uint16:
                        image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            
                    elif image.ndim == 3:  # 8-bit 3D image (likely RGB)
                        image = image[:, :, 0]  # Extract only the first channel
            
                    
                    modifications = np.load(numpy_file)                    
                    mask = np.where(modifications > 0, 1, 0).astype(np.uint8)  # Replace only non-zero pixels
                    
                    # Combine the modified image with the original image using the mask
                    combined_image = np.where(mask == 1, modifications, image).astype(np.uint8)  # Replace only non-zero pixels
                    
                    cv2.imwrite(f"{modified_images_path}{os.path.splitext(image_file)[0].split('/')[-1]}.jpeg", combined_image)

                    