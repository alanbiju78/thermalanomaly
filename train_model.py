import numpy as np
from build_model import build_autoencoder
import pickle # Add this import too, for the history

# --- Load Data ---
train_data = np.load('train_data.npy') # <-- ADD THIS LINE
print("Loaded training data.")

# --- Build and Compile Model ---
autoencoder = build_autoencoder()
autoencoder.compile(optimizer='adam', loss='mean_squared_error')
print("Model compiled.")

# --- Train the Model ---
# Store the training history in a variable
history = autoencoder.fit(
    train_data,
    train_data,
    epochs=150,  # <-- Keep this at 100 (or even 150)
    batch_size=16,
    shuffle=True,
    validation_split=0.1 # This is good practice
)

# --- Save the Trained Model ---
autoencoder.save('thermal_anomaly_model.keras')
print("Model saved as thermal_anomaly_model.keras")

# --- SAVE THE HISTORY ---
with open('train_history.pkl', 'wb') as file_pi:
    pickle.dump(history.history, file_pi)

print("Training history saved as train_history.pkl")