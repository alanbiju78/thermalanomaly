import numpy as np
from build_model import build_autoencoder

# --- Load Data ---
train_data = np.load('train_data.npy')
print("Loaded training data.")

# --- Build and Compile Model ---
autoencoder = build_autoencoder()
autoencoder.compile(optimizer='adam', loss='mean_squared_error')
print("Model compiled.")

# --- Train the Model ---
# Note: The input (x) and the target (y) are the same!
# The model learns to reconstruct its input.
autoencoder.fit(
    train_data,
    train_data,
    epochs=50, # Adjust as needed
    batch_size=16,
    shuffle=True
)

# --- Save the Trained Model ---
autoencoder.save('thermal_anomaly_model.h5')
print("Model saved as thermal_anomaly_model.h5")