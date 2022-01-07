import tensorflow as tf
import tensorflow.keras as keras
from matplotlib import pyplot

filepath = 'trained_generator'
latent_dim = 128 # same as that used in training
num_examples = 50

#load model
model = keras.models.load_model(filepath)

# generate images 
noise = tf.random.normal([num_examples, latent_dim])
images = model(noise)

# plot images
for i in range(1,50):
    pyplot.subplot(5, 10, i)
    pyplot.axis('off')
    pyplot.imshow(images[i, :, :, 0], cmap="gray_r")
pyplot.show()

