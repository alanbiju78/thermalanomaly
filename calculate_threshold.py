import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

# Load the model and the training data
model = load_model('thermal_anomaly_model.keras')
train_data = np.load('train_data.npy')

# Get reconstructions for the training data
reconstructions = model.predict(train_data)

# Calculate the Mean Squared Error (MSE) for each image
train_loss = np.mean(np.square(reconstructions - train_data), axis=(1, 2, 3))

# Plot the distribution of reconstruction errors
plt.hist(train_loss, bins=50)
plt.xlabel("Reconstruction Error (MSE)")
plt.ylabel("Number of images")
plt.title("Distribution of Errors on Normal Data")
plt.show()

# --- Determine Threshold ---
# A good starting point is the mean error plus a number of standard deviations
threshold = np.mean(train_loss) + 2.5 * np.std(train_loss)
print(f"Calculated Anomaly Threshold: {threshold}")

# Save the threshold value
np.save('anomaly_threshold.npy', np.array([threshold]))