from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Conv2DTranspose
from tensorflow.keras.models import Model

def build_autoencoder(input_shape=(128, 128, 1)):
    # --- Encoder ---
    # Input -> 128x128x1
    input_img = Input(shape=input_shape)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
    x = MaxPooling2D((2, 2), padding='same')(x) # -> 64x64x32
    x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), padding='same')(x) # -> 32x32x16
    x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    encoded = MaxPooling2D((2, 2), padding='same')(x) # -> 16x16x8 (bottleneck)

    # --- Decoder ---
    # 16x16x8
    x = Conv2DTranspose(8, (3, 3), activation='relu', padding='same')(encoded)
    x = UpSampling2D((2, 2))(x) # -> 32x32x8
    x = Conv2DTranspose(16, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x) # -> 64x64x16
    x = Conv2DTranspose(32, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x) # -> 128x128x32
    decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x) # -> 128x128x1

    # --- Autoencoder Model ---
    autoencoder = Model(input_img, decoded)
    return autoencoder

if __name__ == '__main__':
    model = build_autoencoder()
    model.summary() # Print model architecture