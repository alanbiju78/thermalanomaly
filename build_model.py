from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Conv2DTranspose
from tensorflow.keras.layers import RandomFlip, RandomRotation, RandomBrightness
from tensorflow.keras.models import Model, Sequential

def build_autoencoder(input_shape=(128, 128, 1)):
    # --- Data Augmentation Layer (Gentler) ---
    data_augmentation = Sequential([
        RandomFlip("horizontal"),
        RandomRotation(0.05),  # Reduced from 0.1
        RandomBrightness(0.1) # Reduced from 0.2
    ], name="augmentation")

    # --- Encoder (Shallower) ---
    input_img = Input(shape=input_shape)
    x = data_augmentation(input_img)
    
    x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), padding='same')(x) # -> 64x64x16
    
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), padding='same')(x) # -> 32x32x32
    
    encoded = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), padding='same')(x) # -> 16x16x64 (Bottleneck)

    # --- Decoder (Shallower) ---
    x = Conv2DTranspose(64, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    
    x = Conv2DTranspose(32, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)

    x = Conv2DTranspose(16, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    
    decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

    autoencoder = Model(input_img, decoded)
    return autoencoder

if __name__ == '__main__':
    model = build_autoencoder()
    model.summary()