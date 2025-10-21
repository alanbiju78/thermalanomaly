import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import smtplib
from email.message import EmailMessage
import mimetypes


# --- Load Model and Threshold ---
MODEL_PATH = 'thermal_anomaly_model.keras'
THRESHOLD_PATH = 'anomaly_threshold.npy'

model = load_model(MODEL_PATH)
threshold = np.load(THRESHOLD_PATH)[0]
IMG_SIZE = (128, 128)

def preprocess_image(img_path):
    """Loads and preprocesses a single image."""
    # Load image in color to support BMP or any format
    img_color = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if img_color is None:
        raise FileNotFoundError(f"Image not found or unreadable at {img_path}")

    # Convert to grayscale as was done during training
    img_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
    
    # Resize and normalize
    img_resized = cv2.resize(img_gray, IMG_SIZE)
    img_normalized = img_resized.astype('float32') / 255.0
    img_final = np.expand_dims(img_normalized, axis=(0, -1))  # Shape: (1, 128, 128, 1)

    # Return both preprocessed input and original for visualization
    original_viz = cv2.resize(img_color, IMG_SIZE)  # Resize original color image for display
    return img_final, original_viz


def visualize_anomaly(original, reconstructed, threshold):
    """Creates a visualization of the anomaly."""
    # Convert reconstructed image to grayscale if not already
    if len(reconstructed.shape) == 3 and reconstructed.shape[-1] == 1:
        reconstructed = reconstructed[:, :, 0]

    # Calculate difference map
    diff = np.abs(original[:, :, 0].astype('float32') - reconstructed.astype('float32'))

    # Create a heatmap of the difference
    heatmap = cv2.applyColorMap((diff * 255).astype(np.uint8), cv2.COLORMAP_JET)

    # Threshold the difference to get a mask of the anomaly
    _, anomaly_mask = cv2.threshold(diff, threshold * 255, 1, cv2.THRESH_BINARY)
    anomaly_mask = (anomaly_mask * 255).astype(np.uint8)

    # Overlay heatmap on original image
    superimposed_img = cv2.addWeighted(heatmap, 0.6, original, 0.4, 0)
    
    return diff, heatmap, superimposed_img

# --- Main Detection Function ---
def check_image_for_anomaly(image_path):
    """
    Checks a single image for anomalies and displays the results.
    """
    try:
        test_img_processed, original_display_img = preprocess_image(image_path)
    except FileNotFoundError as e:
        print(e)
        return

    # Get the model's reconstruction
    reconstruction = model.predict(test_img_processed)[0]

    # Calculate the reconstruction error
    error = np.mean(np.square(reconstruction - test_img_processed))
    
    print(f"Image: {image_path}")
    print(f"Reconstruction Error: {error}")
    print(f"Anomaly Threshold: {threshold}")
    
    if error > threshold:
        print("🚨 Anomaly Detected!")
        
        
        # Prepare images for display
        reconstructed_img_viz = (reconstruction * 255).astype(np.uint8)
        original_gray_for_viz = (test_img_processed[0] * 255).astype(np.uint8)

        diff, heatmap, overlay = visualize_anomaly(original_display_img, reconstructed_img_viz, 0.2)

        # Display results
        fig, axs = plt.subplots(1, 4, figsize=(20, 5))
        axs[0].imshow(cv2.cvtColor(original_display_img, cv2.COLOR_BGR2RGB))
        axs[0].set_title('Original Image')
        axs[0].axis('off')

        axs[1].imshow(reconstructed_img_viz[:, :, 0], cmap='gray')
        axs[1].set_title('Reconstructed Image')
        axs[1].axis('off')

        axs[2].imshow(heatmap, cmap='jet')
        axs[2].set_title('Difference Heatmap')
        axs[2].axis('off')

        axs[3].imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
        axs[3].set_title('Anomaly Highlighted')
        axs[3].axis('off')

        plt.show()

        _, jpeg = cv2.imencode('.jpg', overlay)
        image_bytes = jpeg.tobytes()
        # Email details
        sender_email = '1@gmail.com'#ue sender email
        receiver_email = '2@gmail.com'#use receiver email
        subject = 'You have a message'
        body = 'This is an email sent from Python with an image attached.'

        # Create the email message
        msg = EmailMessage()
        msg['Subject'] = subject
        msg['From'] = sender_email
        msg['To'] = receiver_email
        msg.set_content(body)

        # Add image attachment
        ##image_path = 'image.jpg'  # Change this to your image path

        # Guess the MIME type of the image
        mime_type, _ = mimetypes.guess_type(image_path)
        mime_type, mime_subtype = mime_type.split('/')

        #with open(image_path, 'rb') as img:
        msg.add_attachment(image_bytes,
                maintype='image',
                subtype='jpeg',
                filename='anomaly_overlay.jpg')  # Optional custom name

        # Send the email
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
            smtp.login(sender_email, '')  # Use Gmail App Password here
            smtp.send_message(msg)

    else:
        print("✅ No Anomaly Detected.")
        

# --- To run the demo ---
# --- To run the demo ---
if __name__ == '__main__':
    
    # 💡 PUT THE PATH TO YOUR SINGLE TEST IMAGE HERE
    IMAGE_TO_TEST = 'data/test/anomalous_image.bmp' 

    print(f"--- Analyzing image: {IMAGE_TO_TEST} ---")
    check_image_for_anomaly(IMAGE_TO_TEST)
