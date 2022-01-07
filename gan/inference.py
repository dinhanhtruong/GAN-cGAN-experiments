import tensorflow as tf
import tensorflow.keras as keras
from matplotlib import pyplot

filepath = 'trained_generator5'
latent_dim = 100 # same as that used in training
num_examples = 50

#load model
model = keras.models.load_model(filepath)

# # generate images 
noise = tf.random.normal([num_examples, latent_dim])
images = model(noise)

pyplot.figure(figsize=(15,10))
# plot images
for i in range(1,50):
    pyplot.subplot(5, 10, i)
    pyplot.axis('off')
    pyplot.imshow(images[i, :, :, 0], cmap="gray_r")
pyplot.show()


# interpolate
noise1 = tf.random.normal([1, latent_dim])
noise2 = tf.random.normal([1, latent_dim])
image1 = model(noise1) 
image2 = model(noise2)

steps = 15
pyplot.figure(figsize=(15,10))
for i in range(steps):
    interp = (i*image1 + ((steps-1)-i)*image2)/(steps-1)
    pyplot.subplot(1,steps, i+1)
    pyplot.axis('off')
    pyplot.imshow(interp[0, :, :, 0], cmap="gray_r")
pyplot.show()