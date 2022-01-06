import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.python.keras.layers.convolutional import Conv2DTranspose
from tensorflow.python.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, LeakyReLU, Flatten, Dense, Dropout, Reshape

def discriminator(input_shape=(28,28,1)):
    model = Sequential([
        # conv downsampling 
        Conv2D(128, (3,3), (2,2), 'same', input_shape=input_shape),
        LeakyReLU(alpha=0.2),
        Conv2D(128, (3,3), strides=(2,2), padding='same'),
        LeakyReLU(alpha=0.2),
        # classifier
        Flatten(), 
        Dropout(0.4),
        Dense(1, 'sigmoid')
    ])
    return model
    

def generator():
    dense_size = 128 * 7 * 7 #we want 128 channels of size 7x7 => project latent vect
    model = Sequential([
        # project for upsampling
        Dense(dense_size),
        LeakyReLU(alpha=0.2),
        Reshape([7,7,128]),
        # upsample (14x14)
        Conv2DTranspose(filters=128, kernel_size=(4,4), strides=(2,2), padding='same'),
        LeakyReLU(alpha=0.2),
        # upsample (28x28)
        Conv2DTranspose(filters=128, kernel_size=(4,4), strides=(2,2), padding='same'),
        LeakyReLU(alpha=0.2),
        # generate single-channel 28x28 img 
        Conv2D(1, (7,7), activation='tanh', padding='same')
    ])
    return model
