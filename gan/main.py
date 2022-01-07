import tensorflow as tf
import tensorflow.keras as keras
import gan
from matplotlib import pyplot

latent_dim = 100
batch_sz = 64
epochs = 3

generator = gan.generator()
discriminator = gan.discriminator()


loss_func = keras.losses.BinaryCrossentropy()
d_optimizer = keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5)
g_optimizer = keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5)

# use ALL images from dataset instead of splitting
(x_train, _), (x_test, _) = keras.datasets.fashion_mnist.load_data()
combined_imgs = tf.concat([x_train, x_test], axis=0)
# normalize  to [-1,1] and reshape to [batch, h,w, 1] (for conv2D)
combined_imgs = (tf.cast(combined_imgs, tf.float32) - 127.5) / 127.5
combined_imgs = tf.reshape(combined_imgs, [-1, 28,28,1])
# make TF dataset obj, shuffle, and batch
dataset = tf.data.Dataset.from_tensor_slices(combined_imgs)
dataset = dataset.shuffle(buffer_size=1024).batch(batch_sz, drop_remainder=True)



#  ============  training loop for single batch ===============
@tf.function
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
    deceptive_labels = tf.ones([batch_sz, 1]) # labels all 1 (want to fool disc)
    # train generator SEPARATELY (don't touch discrim, no access to real data)
    with tf.GradientTape() as tape:
        predictions = discriminator(generator(noise))
        g_loss = loss_func(deceptive_labels, predictions)
    grads = tape.gradient(g_loss, generator.trainable_variables)
    g_optimizer.apply_gradients(zip(grads, generator.trainable_variables))

    return d_loss, g_loss, fake_imgs


# ==================   main training loop   =====================
G_losses = []
D_losses = []
for epoch in range(epochs):
    print("Epoch: ", epoch)
    # iterate over batches
    for batch_num, real_img_batch in enumerate(dataset):
        print("batch: ", batch_num)
        d_loss, g_loss, fake_imgs = train_batch(real_img_batch)
        if batch_num % 5 == 0:
            print("D loss: ", d_loss)
            print("G loss: ", g_loss)
        G_losses.append(g_loss)
        D_losses.append(d_loss)

    # plot losses per epoch
    pyplot.plot(G_losses, label='generator')
    pyplot.plot(D_losses, label='discriminator')
    pyplot.xlabel("batch")
    pyplot.ylabel("loss")
    pyplot.legend()
    pyplot.title("G vs. D losses per epoch")
    pyplot.show()

# save weights
generator.save("trained_generator6")

