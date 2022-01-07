import tensorflow as tf
import tensorflow.keras as keras
from matplotlib import pyplot

filepath = 'trained_conditional_generator'
latent_dim = 100 # same as that used in training
num_examples = 8
num_classes = 10

#load model
model = keras.models.load_model(filepath)

# generate images for each class
pyplot.figure(figsize=(15,10))
for i in range(num_classes):
    noise = tf.random.normal([num_examples, latent_dim])
    labels = tf.cast(tf.expand_dims(tf.repeat(i, repeats=num_examples), axis=-1), dtype=tf.float32) #[i,i,i,i...,i] shape = [num_examples, 1]
    images = model(noise, labels)

    # plot images
    for j in range(num_examples):
        pyplot.subplot(num_classes, num_examples, num_examples*i + j+1)
        pyplot.axis('off')
        pyplot.imshow(images[j, :, :, 0], cmap="gray_r")
pyplot.show()


# interpolate in latent space in same class
class_label = 1
label = tf.cast(tf.expand_dims([class_label], axis=-1), dtype=tf.float32)

noise1 = tf.random.normal([1, latent_dim])
image1 = model(noise1, label) 
noise2 = tf.random.normal([1, latent_dim])
image2 = model(noise2, label) 

steps = 15
pyplot.figure(figsize=(10,10))
for i in range(steps):
    interp = (i*image1 + ((steps-1)-i)*image2)/(steps-1)
    pyplot.subplot(1,steps, i+1)
    pyplot.axis('off')
    pyplot.imshow(interp[0, :, :, 0], cmap="gray_r")
pyplot.show()