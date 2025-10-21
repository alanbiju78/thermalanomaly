import os
import cv2
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

# --- Configuration ---
MODEL_PATH = 'thermal_anomaly_model.keras'
HISTORY_PATH = 'train_history.pkl'
THRESHOLD_PATH = 'anomaly_threshold.npy'
TRAIN_DATA_PATH = 'train_data.npy' # Used for error distribution
TEST_NORMAL_DIR = 'data/test/normal/'
TEST_ANOMALY_DIR = 'data/test/anomaly/'
IMG_SIZE = (128, 128)

# Load model and threshold
print("Loading model and data...")
model = load_model(MODEL_PATH)
threshold = np.load(THRESHOLD_PATH)[0]
train_data = np.load(TRAIN_DATA_PATH)

# Helper function to load and preprocess images from a directory
def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            img_resized = cv2.resize(img, IMG_SIZE)
            img_normalized = img_resized.astype('float32') / 255.0
            img_final = np.expand_dims(img_normalized, axis=-1)
            images.append(img_final)
    return np.array(images)

# Helper function to get reconstruction error (MSE)
def get_reconstruction_error(model, data):
    reconstructions = model.predict(data)
    # Calculate MSE for each image
    errors = np.mean(np.square(data - reconstructions), axis=(1, 2, 3))
    return errors

## --- 1. Plot Training Loss Curve ---
def plot_training_history(history_path):
    print("Generating Training Loss Plot...")
    with open(history_path, 'rb') as file_pi:
        history = pickle.load(file_pi)
        
    plt.figure(figsize=(10, 6))
    plt.plot(history['loss'], label='Training Loss')
    if 'val_loss' in history:
        plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Model Training History')
    plt.ylabel('Loss (Mean Squared Error)')
    plt.xlabel('Epoch')
    plt.legend()
    plt.grid(True)
    plt.savefig('1_training_loss_curve.png')
    print("Saved '1_training_loss_curve.png'")
    # plt.show()

## --- 2. Plot Reconstruction Error Distribution ---
def plot_error_distribution(model, train_data, anomaly_data, threshold):
    print("Generating Error Distribution Plot...")
    normal_errors = get_reconstruction_error(model, train_data)
    anomaly_errors = get_reconstruction_error(model, anomaly_data)

    plt.figure(figsize=(12, 7))
    sns.histplot(normal_errors, bins=50, kde=True, color='blue', label='Normal (Healthy)')
    sns.histplot(anomaly_errors, bins=50, kde=True, color='red', label='Anomalous (Fault)')
    
    plt.axvline(threshold, 0, 1, color='black', linestyle='--', label=f'Threshold = {threshold:.4f}')
    
    plt.title('Reconstruction Error Distribution')
    plt.xlabel('Reconstruction Error (MSE)')
    plt.ylabel('Number of Images')
    plt.legend()
    plt.savefig('2_error_distribution.png')
    print("Saved '2_error_distribution.png'")
    # plt.show()

## --- 3. Plot Reconstruction Examples ---
def plot_reconstruction_examples(model, normal_img, anomaly_img):
    print("Generating Reconstruction Example Plot...")
    # Get reconstructions
    recon_normal = model.predict(np.expand_dims(normal_img, axis=0))[0]
    recon_anomaly = model.predict(np.expand_dims(anomaly_img, axis=0))[0]

    fig, axs = plt.subplots(2, 2, figsize=(8, 8))
    
    # Normal Image
    axs[0, 0].imshow(normal_img.squeeze(), cmap='gray')
    axs[0, 0].set_title('Original Normal Image')
    axs[0, 0].axis('off')
    
    axs[0, 1].imshow(recon_normal.squeeze(), cmap='gray')
    axs[0, 1].set_title('Reconstructed Normal Image')
    axs[0, 1].axis('off')

    # Anomaly Image
    axs[1, 0].imshow(anomaly_img.squeeze(), cmap='gray')
    axs[1, 0].set_title('Original Anomalous Image')
    axs[1, 0].axis('off')
    
    axs[1, 1].imshow(recon_anomaly.squeeze(), cmap='gray')
    axs[1, 1].set_title('Reconstructed Anomalous Image')
    axs[1, 1].axis('off')

    plt.tight_layout()
    plt.savefig('3_reconstruction_examples.png')
    print("Saved '3_reconstruction_examples.png'")
    # plt.show()
    
    return recon_normal, recon_anomaly

## --- 4. Plot Anomaly Heatmaps ---
def plot_heatmaps(original_normal, recon_normal, original_anomaly, recon_anomaly):
    print("Generating Anomaly Heatmap Plot...")
    # Calculate difference
    diff_normal = np.abs(original_normal.squeeze() - recon_normal.squeeze())
    diff_anomaly = np.abs(original_anomaly.squeeze() - recon_anomaly.squeeze())

    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    
    axs[0].imshow(diff_normal, cmap='jet')
    axs[0].set_title('Difference Map (Normal)')
    axs[0].axis('off')
    
    im = axs[1].imshow(diff_anomaly, cmap='jet')
    axs[1].set_title('Difference Map (Anomaly)')
    axs[1].axis('off')
    
    fig.colorbar(im, ax=axs[1], orientation='vertical')
    plt.tight_layout()
    plt.savefig('4_anomaly_heatmaps.png')
    print("Saved '4_anomaly_heatmaps.png'")
    # plt.show()

## --- 5. Plot Confusion Matrix & Report ---
def plot_confusion_matrix_and_report(model, normal_data, anomaly_data, threshold):
    print("Generating Confusion Matrix and Report...")
    # Get errors
    normal_errors = get_reconstruction_error(model, normal_data)
    anomaly_errors = get_reconstruction_error(model, anomaly_data)
    
    # Create true labels
    # 0 = Normal, 1 = Anomaly
    y_true_normal = [0] * len(normal_errors)
    y_true_anomaly = [1] * len(anomaly_errors)
    y_true = y_true_normal + y_true_anomaly
    
    # Create predicted labels
    y_pred_normal = [1 if e > threshold else 0 for e in normal_errors]
    y_pred_anomaly = [1 if e > threshold else 0 for e in anomaly_errors]
    y_pred = y_pred_normal + y_pred_anomaly
    
    # --- Print Classification Report ---
    report = classification_report(y_true, y_pred, target_names=['Normal (Class 0)', 'Anomaly (Class 1)'])
    print("\n--- Classification Report ---")
    print(report)
    
    # --- Plot Confusion Matrix ---
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Normal', 'Anomaly'])
    
    disp.plot(cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.savefig('5_confusion_matrix.png')
    print("Saved '5_confusion_matrix.png'")
    # plt.show()

# --- Main execution ---
if __name__ == '__main__':
    # Load test data
    test_normal_data = load_images_from_folder(TEST_NORMAL_DIR)
    test_anomaly_data = load_images_from_folder(TEST_ANOMALY_DIR)
    
    if len(test_normal_data) == 0 or len(test_anomaly_data) == 0:
        print(f"Error: Could not find images in {TEST_NORMAL_DIR} or {TEST_ANOMALY_DIR}.")
        print("Please create these folders and add test images.")
    else:
        # --- Run all plotting functions ---
        
        # 1. Plot loss
        plot_training_history(HISTORY_PATH)
        
        # 2. Plot error distribution
        plot_error_distribution(model, train_data, test_anomaly_data, threshold)

        # Get images for examples
        example_normal = test_normal_data[0]
        example_anomaly = test_anomaly_data[0]

        # 3. Plot reconstructions
        recon_norm, recon_anom = plot_reconstruction_examples(model, example_normal, example_anomaly)
        
        # 4. Plot heatmaps
        plot_heatmaps(example_normal, recon_norm, example_anomaly, recon_anom)
        
        # 5. Plot confusion matrix
        plot_confusion_matrix_and_report(model, test_normal_data, test_anomaly_data, threshold)

        print("\nAll visualizations saved as PNG files in your project directory.")