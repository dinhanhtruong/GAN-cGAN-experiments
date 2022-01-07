import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.python.ops.gen_batch_ops import batch
import cgan
from matplotlib import pyplot


# ================ hyperparams ================
latent_dim = 100
batch_sz = 64
embedding_sz = 50
epochs = 3
num_classes = 10 #mnist has 10

generator = cgan.cond_generator(num_classes, embedding_sz)
discriminator = cgan.cond_discriminator(num_classes, embedding_sz)

loss_func = keras.losses.BinaryCrossentropy()
d_optimizer = keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5)
g_optimizer = keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5)


# ================ data preprocessing ============
# use ALL images from dataset instead of splitting
(x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()
combined_imgs = tf.concat([x_train, x_test], axis=0)
# normalize  to [-1,1] 
combined_imgs = (tf.cast(combined_imgs, tf.float32) - 127.5) / 127.5
combined_imgs = tf.reshape(combined_imgs, [-1, 28,28,1])
# combine labels in same order
combined_labels = tf.cast(tf.concat([y_train, y_test], axis=0), tf.float32)
combined_labels = tf.expand_dims(combined_labels, axis=-1)
# make TF dataset obj, shuffle, and batch
dataset = tf.data.Dataset.from_tensor_slices((combined_imgs, combined_labels))
dataset = dataset.shuffle(buffer_size=1024).batch(batch_sz, drop_remainder=True)



#  ============  training loop for single batch ===============
@tf.function
def train_batch(real_imgs, real_img_classes): # batch of real imgs
    # sample random latent vects from N(0,1)
    noise = tf.random.normal([batch_sz, latent_dim])
    # sample random class from 0-9 to be fed into generator
    rand_classes = tf.random.uniform([batch_sz, 1], 0, 9, dtype=tf.int32)
    rand_classes = tf.cast(rand_classes, tf.float32) # convert to float32
    # generate fake images
    fake_imgs = generator(noise, rand_classes)
    # combine fake and real to be fed into discrimiantor [batch_sz*2, 28,28,1], and make discrim labels (1=real, 0=fake)
    discriminator_img_input = tf.concat([real_imgs, fake_imgs], axis=0)
    true_labels = tf.concat([tf.ones([real_imgs.shape[0], 1]), tf.zeros([batch_sz, 1])], axis=0)
    # combine conditional 0-9 labels
    conditional_class_labels = tf.concat([real_img_classes, rand_classes], axis=0)
    # train discriminator
    with tf.GradientTape() as tape:
        predictions = discriminator(discriminator_img_input, conditional_class_labels)
        d_loss = loss_func(true_labels, predictions)
    grads = tape.gradient(d_loss, discriminator.trainable_variables)
    d_optimizer.apply_gradients(zip(grads, discriminator.trainable_variables))


    # sample points for generator training with deceptive labels (1=fake) and ALL generated fake images are 'real' in disc's perspective
    noise = tf.random.normal([batch_sz, latent_dim])
    rand_classes = tf.random.uniform([batch_sz, 1], 0, 9, dtype=tf.int32)
    deceptive_labels = tf.ones([batch_sz, 1]) # labels all 1 (want to fool disc)
    # train generator SEPARATELY (don't touch discrim, no access to real data)
    with tf.GradientTape() as tape:
        predictions = discriminator(generator(noise, rand_classes), rand_classes)
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
    for batch_num, img_label_pair in enumerate(dataset):
        real_img_batch = img_label_pair[0]
        real_label_batch = img_label_pair[1]
        d_loss, g_loss, fake_imgs = train_batch(real_img_batch, real_label_batch)
        if batch_num % 5 == 0:
            print("Batch: ", batch_num)
            print("D loss: ", d_loss)
            print("G loss: ", g_loss)
        G_losses.append(g_loss)
        D_losses.append(d_loss)

        # plot losses 
        if batch_num % 20 == 0:
            pyplot.plot(G_losses, label='generator')
            pyplot.plot(D_losses, label='discriminator')
            pyplot.xlabel("batch")
            pyplot.ylabel("loss")
            pyplot.legend()
            pyplot.title("G vs. D losses")
            pyplot.show()

# save weights
generator.save("trained_conditional_generator")

