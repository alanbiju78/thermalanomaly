import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model, Model
from sklearn.metrics import precision_recall_curve, auc, roc_curve, roc_auc_score
from sklearn.manifold import TSNE

# --- Configuration ---
MODEL_PATH = 'thermal_anomaly_model.keras'
TEST_NORMAL_DIR = 'data/test/normal/'
TEST_ANOMALY_DIR = 'data/test/anomaly/'
IMG_SIZE = (128, 128)

print("Loading model and data...")
# Load the full autoencoder model
autoencoder = load_model(MODEL_PATH)

# --- Create a separate ENCODER model ---
# We need to find the "bottleneck" layer. 
# Based on the model in Step 2, 'conv2d_2' is the 3rd Conv layer
# and 'max_pooling2d_2' is the final encoding layer.
# Let's use the name of the layer before the bottleneck.
# You can check your model's layer names with `autoencoder.summary()`
# For the model we built, the bottleneck layer is named 'max_pooling2d_2'
try:
    encoder = Model(inputs=autoencoder.input, 
                    outputs=autoencoder.get_layer('max_pooling2d_2').output)
except ValueError as e:
    print(f"Error creating encoder model: {e}")
    print("Please check the bottleneck layer name in `autoencoder.summary()`")
    print("Using 'encoded' as a fallback if you named it.")
    # Fallback if you followed an older guide and named the layer
    try:
        encoder = Model(inputs=autoencoder.input, 
                        outputs=autoencoder.get_layer('encoded').output)
    except:
        print("Could not find bottleneck layer. Exiting.")
        exit()


# Helper function to load and preprocess images
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
    errors = np.mean(np.square(data - reconstructions), axis=(1, 2, 3))
    return errors

# --- Load all test data ---
normal_data = load_images_from_folder(TEST_NORMAL_DIR)
anomaly_data = load_images_from_folder(TEST_ANOMALY_DIR)

# Combine for analysis
all_test_data = np.concatenate((normal_data, anomaly_data))

# --- Create True Labels ---
# 0 = Normal, 1 = Anomaly
y_true_normal = [0] * len(normal_data)
y_true_anomaly = [1] * len(anomaly_data)
y_true = np.array(y_true_normal + y_true_anomaly)

# --- Get Reconstruction Errors (Scores) ---
# The error score is our model's "prediction"
errors = get_reconstruction_error(autoencoder, all_test_data)

## --- 1. Generate Precision-Recall (PR) Curve ---
def plot_pr_curve(y_true, errors):
    print("Generating Precision-Recall Curve...")
    precision, recall, _ = precision_recall_curve(y_true, errors)
    pr_auc = auc(recall, precision)

    plt.figure(figsize=(10, 7))
    plt.plot(recall, precision, color='blue', lw=2, label=f'PR curve (AUC = {pr_auc:0.2f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc='lower left')
    plt.grid(True)
    plt.savefig('6_precision_recall_curve.png')
    print("Saved '6_precision_recall_curve.png'")
    # plt.show()

## --- 2. Generate ROC Curve ---
def plot_roc_curve(y_true, errors):
    print("Generating ROC Curve...")
    fpr, tpr, _ = roc_curve(y_true, errors)
    roc_auc = roc_auc_score(y_true, errors)

    plt.figure(figsize=(10, 7))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:0.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.savefig('7_roc_curve.png')
    print("Saved '7_roc_curve.png'")
    # plt.show()

## --- 3. Generate t-SNE Visualization ---
def plot_tsne(encoder_model, normal_data, anomaly_data):
    print("Generating t-SNE plot (this may take a minute)...")
    # Get the "encoded" representations (latent space)
    normal_encoded = encoder_model.predict(normal_data)
    anomaly_encoded = encoder_model.predict(anomaly_data)
    
    # Flatten the encoded vectors for t-SNE
    nsamples, nx, ny, nz = normal_encoded.shape
    normal_encoded_flat = normal_encoded.reshape((nsamples, nx*ny*nz))
    
    asamples, ax, ay, az = anomaly_encoded.shape
    anomaly_encoded_flat = anomaly_encoded.reshape((asamples, ax*ay*az))

    all_encoded_flat = np.concatenate((normal_encoded_flat, anomaly_encoded_flat))
    
    # Create labels for t-SNE plot
    labels = ['Normal'] * len(normal_encoded_flat) + ['Anomaly'] * len(anomaly_encoded_flat)
    
    # Run t-SNE
    tsne = TSNE(n_components=2, verbose=1, perplexity=40, max_iter=300)
    tsne_results = tsne.fit_transform(all_encoded_flat)
    
    # Plot the results
    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(tsne_results[:,0], tsne_results[:,1], c=[0 if l == 'Normal' else 1 for l in labels], cmap='coolwarm', alpha=0.7)
    plt.title('t-SNE Visualization of Latent Space')
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    
    # Create a legend
    handles, _ = scatter.legend_elements()
    plt.legend(handles=handles, labels=['Normal', 'Anomaly'])
    
    plt.grid(True)
    plt.savefig('8_tsne_visualization.png')
    print("Saved '8_tsne_visualization.png'")
    # plt.show()


# --- Main execution ---
if __name__ == '__main__':
    plot_pr_curve(y_true, errors)
    plot_roc_curve(y_true, errors)
    plot_tsne(encoder, normal_data, anomaly_data)

    print("\nAdvanced visualizations saved as PNG files.")