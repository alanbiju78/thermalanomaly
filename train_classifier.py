import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Rescaling, Dropout
from tensorflow.keras.layers import RandomFlip, RandomRotation, RandomZoom

# --- Configuration ---
IMG_SIZE = (128, 128)
BATCH_SIZE = 32
TRAIN_DIR = 'classifier_data/train'

# --- Load Data ---
train_dataset = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    labels='inferred',
    label_mode='binary',
    image_size=IMG_SIZE,
    validation_split=0.2,
    subset='training',
    seed=123,
    batch_size=BATCH_SIZE,
    color_mode='grayscale'
)

validation_dataset = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    labels='inferred',
    label_mode='binary',
    image_size=IMG_SIZE,
    validation_split=0.2,
    subset='validation',
    seed=123,
    batch_size=BATCH_SIZE,
    color_mode='grayscale'
)

print(f"Found class names: {train_dataset.class_names}")

# --- Build a More Robust CNN Classifier Model ---

# Create a data augmentation block
data_augmentation = Sequential([
    RandomFlip("horizontal"),
    RandomRotation(0.1),
    RandomZoom(0.1),
], name="augmentation")

classifier = Sequential([
    # Input layer
    tf.keras.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 1)),
    
    # Apply data augmentation ONLY to the training images
    data_augmentation,
    
    # Rescale pixel values
    Rescaling(1./255),
    
    # Convolutional blocks
    Conv2D(32, 3, padding='same', activation='relu'),
    MaxPooling2D(),
    
    Conv2D(64, 3, padding='same', activation='relu'),
    MaxPooling2D(),
    
    Conv2D(128, 3, padding='same', activation='relu'),
    MaxPooling2D(),
    
    Flatten(),
    
    # Dense block with Dropout to prevent overfitting
    Dense(128, activation='relu'),
    Dropout(0.5), # Add dropout layer
    
    # Final output neuron for binary classification
    Dense(1, activation='sigmoid') 
])

# --- Compile the Model ---
classifier.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# --- Train the Model ---
print("\n--- Starting Robust Classifier Training ---")
classifier.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=25 # Increased epochs for more robust training
)

# --- Save the Trained Classifier ---
classifier.save('machine_classifier.h5')
print("\nNew, robust classifier model saved as machine_classifier.h5")