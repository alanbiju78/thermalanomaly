import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import smtplib
from email.message import EmailMessage

# --- Configuration & Model Loading ---
IMG_SIZE = (128, 128)
CLASSIFIER_THRESHOLD = 0.5  # Confidence threshold for the classifier

try:
    classifier = tf.keras.models.load_model('machine_classifier.h5')
    autoencoder = tf.keras.models.load_model('thermal_anomaly_model.h5')
    anomaly_threshold = np.load('anomaly_threshold.npy')[0]
    print("✅ All models and thresholds loaded successfully.")
except Exception as e:
    print(f"❌ Error loading models: {e}")
    print("Please ensure 'machine_classifier.h5', 'thermal_anomaly_model.h5', and 'anomaly_threshold.npy' are present.")
    exit()

# --- Helper Functions ---

def preprocess_image(img_path):
    """Loads and preprocesses a single image."""
    # Load image in color to get original for visualization
    img_color = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if img_color is None:
        raise FileNotFoundError(f"Image not found or unreadable at {img_path}")

    # Convert to grayscale for model input
    img_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
    
    # Resize and normalize
    img_resized = cv2.resize(img_gray, IMG_SIZE)
    img_normalized = img_resized.astype('float32') / 255.0
    img_final = np.expand_dims(img_normalized, axis=(0, -1))

    # Return both model input and the color version for visualization
    original_viz = cv2.resize(img_color, IMG_SIZE)
    return img_final, original_viz

def visualize_and_alert(original_display_img, reconstruction, image_path):
    """Handles the visualization and email alert for a detected anomaly."""
    print("-> Result: 🚨 Anomaly Detected!")
    
    # --- Visualization ---
    reconstructed_img_viz = (reconstruction[0] * 255).astype(np.uint8)
    
    # Create the difference map by comparing the grayscale version of the original and the reconstruction
    original_gray_for_diff = cv2.cvtColor(original_display_img, cv2.COLOR_BGR2GRAY)
    diff = cv2.absdiff(original_gray_for_diff, reconstructed_img_viz[:, :, 0])
    
    heatmap = cv2.applyColorMap(diff, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(original_display_img, 0.6, heatmap, 0.4, 0)

    # Display results
    fig, axs = plt.subplots(1, 4, figsize=(20, 5))
    axs[0].imshow(cv2.cvtColor(original_display_img, cv2.COLOR_BGR2RGB))
    axs[0].set_title('Original Image')
    axs[0].axis('off')

    axs[1].imshow(reconstructed_img_viz[:, :, 0], cmap='gray')
    axs[1].set_title('Reconstructed Image')
    axs[1].axis('off')

    axs[2].imshow(heatmap)
    axs[2].set_title('Difference Heatmap')
    axs[2].axis('off')

    axs[3].imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
    axs[3].set_title('Anomaly Highlighted')
    axs[3].axis('off')

    plt.show()

    # --- Email Alert ---
    print("--> Sending email alert...")
    _, jpeg = cv2.imencode('.jpg', overlay)
    image_bytes = jpeg.tobytes()
    
    sender_email = 'biznexprojectmini@gmail.com'
    receiver_email = 'blessonveenus19@gmail.com'
    app_password = 'egqe pdjh gwpi cnhd' # Your Gmail App Password
    
    msg = EmailMessage()
    msg['Subject'] = 'Anomaly Detected in Monitored Machine'
    msg['From'] = sender_email
    msg['To'] = receiver_email
    msg.set_content(f'An anomaly was detected in the machine. Please see the attached image from file: {image_path}')
    msg.add_attachment(image_bytes, maintype='image', subtype='jpeg', filename='anomaly_report.jpg')
    
    try:
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
            smtp.login(sender_email, app_password)
            smtp.send_message(msg)
        print("--> Email sent successfully.")
    except Exception as e:
        print(f"--> Failed to send email: {e}")


# --- Main Detection Function ---

def run_full_check(image_path):
    """Performs the full two-step check: identification then anomaly detection."""
    try:
        processed_image, original_display_img = preprocess_image(image_path)
    except FileNotFoundError as e:
        print(e)
        return

    # === STEP 1: Machine Identification ===
    prediction_score = classifier.predict(processed_image)[0][0]

    # Check class order from your training output. Assuming 'other_machinery' is class 1.
    if prediction_score > CLASSIFIER_THRESHOLD: 
        print(f"-> Result: Machine Uncheckable (Confidence: {prediction_score:.2f})")
        print("-> Reason: Image does not appear to be the target induction motor.")
        return

    print(f"✅ Machine identified as target motor (Confidence: {1-prediction_score:.2f}).")
    print("--- Proceeding to Anomaly Check ---")

    # === STEP 2: Anomaly Detection ===
    reconstruction = autoencoder.predict(processed_image)
    error = np.mean(np.square(reconstruction - processed_image))

    print(f"Reconstruction Error: {error:.6f}")
    print(f"Anomaly Threshold:    {anomaly_threshold:.6f}")

    if error > anomaly_threshold:
        visualize_and_alert(original_display_img, reconstruction, image_path)
    else:
        print("-> Result: ✅ No Anomaly Detected.")


# --- Example Usage ---
if __name__ == '__main__':
    # === Test Case 1: An anomalous image of the CORRECT motor ===
    print("\n--- Checking an anomalous image of the target motor ---")
    correct_motor_anomaly_path = 'data/test/anomalous_image.bmp' # MAKE SURE THIS PATH IS CORRECT
    run_full_check(correct_motor_anomaly_path)

    # === Test Case 2: An image of the WRONG machine ===
    print("\n--- Checking an image of a DIFFERENT machine ---")
    wrong_machine_path = 'data/test/other_machine_image.jpg' # MAKE SURE THIS PATH IS CORRECT
    run_full_check(wrong_machine_path)