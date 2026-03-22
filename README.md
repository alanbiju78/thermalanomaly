# Unsupervised Thermal Anomaly Detection in Induction Motors

This project implements a deep learning pipeline for automated condition monitoring of induction motors using thermal imagery. The system uses a **Convolutional Autoencoder (CAE)** trained exclusively on images of a healthy motor, allowing it to detect and localize unseen thermal anomalies in an unsupervised manner.

This approach is ideal for predictive maintenance, as it does not require collecting a large and diverse dataset of "faulty" images, which is often impractical or dangerous.

*(A sample output showing how the system localizes a fault (right) while finding no error in a normal image (left))*

## 🚀 Project Overview

The core of this project is an unsupervised anomaly detection model.

1.  **Training:** The CAE is trained *only* on thermal images of a healthy, optimally-functioning motor. It learns to compress and then reconstruct these "normal" images with very high fidelity (low reconstruction error).
2.  **Detection:** When the trained model is shown a new image, it attempts to reconstruct it.
      * If the image is **normal**, the reconstruction will be accurate, and the error will be low.
      * If the image is **anomalous** (e.g., a bearing is overheating), the model will fail to reconstruct it accurately, resulting in a **high reconstruction error**.
3.  **Localization:** By subtracting the reconstructed image from the original, we can create a "difference heatmap" that visually pinpoints the exact location of the anomaly.

## ✨ Key Features

  * **Unsupervised:** Detects any deviation from "normal" without ever being trained on fault data.
  * **Anomaly Detection:** Classifies new images as "Normal" or "Anomalous" in real-time.
  * **Anomaly Localization:** Generates heatmaps to show *where* the fault is located on the motor.
  * **Research-Ready:** Includes scripts to automatically generate all graphs and metrics (Loss curves, ROC curves, error distributions, etc.) for a research paper.

## 💻 Tech Stack

  * **Python 3.x**
  * **TensorFlow / Keras:** For building and training the CAE model.
  * **OpenCV (cv2):** For image preprocessing (reading, resizing).
  * **NumPy:** For numerical operations and data handling.
  * **Scikit-learn:** For generating classification reports, confusion matrices, and ROC curves.
  * **Matplotlib / Seaborn:** For plotting all visualizations.

## 🛠️ Installation & Setup

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/your-username/your-repo-name.git
    cd your-repo-name
    ```

2.  **Create a virtual environment (recommended):**

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install the required libraries:**
    A `requirements.txt` file is provided for easy setup.

    ```bash
    pip install -r requirements.txt
    ```

    *If you don't have a `requirements.txt`, you can create one with the following content:*

    ```text
    tensorflow
    opencv-python
    numpy
    matplotlib
    scikit-learn
    seaborn
    ```

## 💾 Data Setup (Crucial Step)

This project uses the "Thermal images of induction motor" dataset, which is publicly available on Mendeley Data.

1.  **Download the dataset:**

      * **Link:** [https://data.mendeley.com/datasets/z3y2z24n62/2](https://www.google.com/search?q=https://data.mendeley.com/datasets/z3y2z24n62/2)
      * Download and unzip the file. You will see 11 folders.

2.  **Organize the data:**

      * The folder `Noload` contains the **25 "Healthy"** images.
      * The other 10 folders (`A&B50`, `Fan`, `Rotor-0`, etc.) contain the **334 "Anomalous"** images.

3.  **Create the project data structure:**
    You must split the data into `train` and `test` sets as follows. This project's success depends on the model *never* seeing the test images during training.

    ```
    your-repo-name/
    ├── data/
    │   ├── train/
    │   │   └── normal/
    │   │       ├── (Place 22 of the 'Noload' images here)
    │   │
    │   └── test/
    │       ├── normal/
    │       │   ├── (Place the remaining 3 'Noload' images here)
    │       │
    │       └── anomaly/
    │           ├── (Place all 334 images from the 10 fault folders here)
    │
    ├── preprocess.py
    ├── build_model.py
    ├── train_model.py
    ├── ... (all other .py files)
    ```

## 🚀 How to Run the Project

Run the scripts from your terminal in this order.

### 1\. Preprocess the Training Data

This script will take the 22 images from `data/train/normal/`, process them, and save them as a single `train_data.npy` file.

```bash
python preprocess.py
```

### 2\. Train the Model

This will build the CAE, train it on `train_data.npy`, and save the final model as `thermal_anomaly_model.keras`. It will also save the training history.

```bash
python train_model.py
```

### 3\. Calculate the Anomaly Threshold

This script loads the trained model, calculates its reconstruction errors on the training data, and saves a statistical threshold to `anomaly_threshold.npy`.

```bash
python calculate_threshold.py
```

### 4\. Run Live Anomaly Detection

This script will load the trained model and the threshold to analyze a new, single image.

**Note:** You must edit `detect_anomaly.py` and change the `IMAGE_TO_TEST` variable to the path of the image you want to test.

```bash
python detect_anomaly.py
```

**Example Output:**

```
--- Analyzing image: data/test/anomaly/Fan/fan_1.bmp ---
Reconstruction Error: 0.0215
Anomaly Threshold: 0.0074
🚨 Anomaly Detected!
```

### 5\. (Optional) Generate Research Visuals

These scripts will load the full test set (both normal and anomaly) and generate all the graphs for your research paper.

```bash
python generate_visuals.py
python generate_advanced_visuals.py
```

This will save 8 PNG files to your directory, such as:

  * `1_training_loss_curve.png`
  * `2_error_distribution.png`
  * `5_confusion_matrix.png`
  * `7_roc_curve.png`
  * ...and more.

## 📊 Example Results

The model clearly learns to separate the two classes, as shown by the distribution of reconstruction errors.

*Fig. 1: Histogram of reconstruction errors. The model's errors for normal images (blue) are clearly distinct from its errors for anomalous images (red), allowing for effective thresholding.*

## 📜 License

This project is licensed under the MIT License. See the `LICENSE` file for details.

## 🙏 Acknowledgments

This project would not be possible without the public dataset provided by the following researchers:

* L. (2020), "Thermal images of induction motor," Mendeley Data, V2, doi: 10.17632/z3y2z24n62.2*
