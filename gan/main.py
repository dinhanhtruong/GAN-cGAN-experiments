import tensorflow as tf
import tensorflow.keras as keras
from matplotlib import pyplot
from tensorflow.python.keras import optimizers
import gan

latent_dim = 128
batch_sz = 128
epochs = 3
generator = gan.generator
discriminator = gan.discriminator
loss_func = keras.losses.BinaryCrossentropy(from_logits=True)
d_optimizer = keras.optimizers.Adam(0.003)
g_optimizer = keras.optimizers.Adam(0.004)

# use ALL images from dataset instead of splitting
(x_train, _), (x_test, _) = keras.datasets.fashion_mnist.load_data()
combined_imgs = tf.concat([x_train, x_test], axis=0)
# normalize and reshape (for conv2D)
combined_imgs /= 255.0
combined_imgs = tf.reshape(combined_imgs, [-1, 28,28,1])
# make TF dataset obj and batch
dataset = tf.data.Dataset.from_tensor_slices(combined_imgs)
dataset = dataset.batch(batch_sz, drop_remainder=True)

def train_batch(real_imgs): # batch of real imgs
    # sample random latent vects from N(0,1)
    noise = tf.random.normal([batch_sz, latent_dim])
    # generate fake images
    fake_imgs = generator(noise)
    # combine fake and real to be fed into discrimiantor [batch_sz*2, 28,28], and make discrim labels (1=real, 0=fake)
    discriminator_in = tf.concat([real_imgs, fake_imgs], axis=0)
    true_labels = tf.concat([tf.ones([real_imgs.shape[0], 1]), tf.zeros([batch_sz, 1])], axis=0)
    # train discriminator
    with tf.GradientTape() as tape:
        predictions = discriminator(discriminator_in)
        d_loss = loss_func(true_labels, predictions)
    grads = tape.gradient(d_loss, discriminator.trainable_variables)
    d_optimizer.apply_gradients(zip(grads, discriminator.trainable_variables))


    # sample points for generator training with deceptive labels (1=fake) and ALL generated fake images are 'real' in disc's perspective
    noise = tf.random.normal([batch_sz, latent_dim])
    fake_imgs = generator(noise)
    deceptive_labels = tf.ones([batch_sz, 1]) # labels all 1 (want to fool disc)
    # train generator SEPARATELY (don't touch discrim, no access to real data)
    with tf.GradientTape() as tape:
        predictions = discriminator(fake_imgs)
        g_loss = loss_func(deceptive_labels, predictions)
    grads = tape.gradient(g_loss, generator.trainable_variables)
    g_optimizer.apply_gradients(zip(grads, generator.trainable_variables))

    return d_loss, g_loss, fake_imgs


# main training loop

# save weights
generator.save("trained_generator")


# generate new images 
# for i in range(1,50):
#     pyplot.subplot(5, 10, i)
#     pyplot.axis('off')
#     pyplot.imshow(x_train[i], cmap="gray_r")
# pyplot.show()