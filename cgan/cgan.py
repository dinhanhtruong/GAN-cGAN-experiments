import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.python.keras.layers import embeddings
from tensorflow.python.keras.layers.convolutional import Conv2DTranspose
from tensorflow.python.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, LeakyReLU, Flatten, Dense, Dropout, Reshape, Embedding



# def cond_discriminator(input_shape=(28,28,1)):
#     model = Sequential([
#         # conv downsampling 
#         Conv2D(128, (3,3), (2,2), 'same', input_shape=input_shape),
#         LeakyReLU(alpha=0.2),
#         Conv2D(128, (3,3), strides=(2,2), padding='same'),
#         LeakyReLU(alpha=0.2),
#         # classifier
#         Flatten(), 
#         Dropout(0.4),
#         Dense(1, activation='sigmoid')
#     ])
#     return model

class cond_discriminator(keras.Model):
    def __init__(self, num_classes, emb_sz):
        super(cond_discriminator, self).__init__()

        # embedding to convert given class labels into dense vectors for conditioning generator output
        self.emb = Embedding(num_classes, emb_sz, input_length=1)
        self.emb_dense = Dense(784)# want 28^2 = 784 pixels for images
        
        self.model = Sequential([
            # conv downsampling 
            Conv2D(128, (3,3), (2,2), 'same'),
            LeakyReLU(alpha=0.2),
            Conv2D(128, (3,3), strides=(2,2), padding='same'),
            LeakyReLU(alpha=0.2),
            # classifier
            Flatten(), 
            Dropout(0.4),
            Dense(1, activation='sigmoid')
        ])

    def call(self, imgs, class_labels):
        """
        class_labels: [batch, 1]
        imgs: [batch, 28, 28, 1]
        returns: scalar [batch, 1]
        """
        # get embedding of label
        embeddings = self.emb(class_labels) #[1, emb_sz]
        # project embedding same dim as image input and reshape from [1, 784] to [28,28,1]
        label_input = self.emb_dense(embeddings) 
        label_input = tf.reshape(label_input, [-1, 28,28,1]) 

        # concat imgs with projected label (as an additional channel => 2 total channels)
        discrim_input = tf.concat([imgs, label_input], axis=3) # channel  axis = 3
        # model forward pass (same as unconditional conv2D)
        return self.model(discrim_input)



class cond_generator(keras.Model):
    def __init__(self, num_classes, emb_sz):
        super(cond_generator, self).__init__()
        single_channel_size = 7 * 7

        # embedding to convert given class labels into dense vectors for conditioning generator output
        self.emb = Embedding(num_classes, emb_sz, input_length=1)
        self.emb_dense = Dense(single_channel_size)
        self.transform_latent_vecs = Sequential([
            # project for upsampling
            Dense(128*single_channel_size), # want 128 channels
            LeakyReLU(alpha=0.2),
            Reshape([7,7,128]),
        ])
        self.model = Sequential([
            # upsample (14x14)
            Conv2DTranspose(filters=128, kernel_size=(4,4), strides=(2,2), padding='same'),
            LeakyReLU(alpha=0.2),
            # upsample (28x28)
            Conv2DTranspose(filters=128, kernel_size=(4,4), strides=(2,2), padding='same'),
            LeakyReLU(alpha=0.2),
            # generate single-channel 28x28 img 
            Conv2D(1, (7,7), activation='tanh', padding='same')
        ])

    def call(self, latent_vecs, class_labels):
        """
        class_labels: [batch, 1]
        latent_vecs: [batch, latent_dim]
        returns: fake image [batch, 28, 28, 1]
        """
        # get embedding of label
        embeddings = self.emb(class_labels) #[1, emb_sz]
        # project embedding same dim as latent vec and reshape from [1, 49] to [7,7,1]
        label_input = self.emb_dense(embeddings)
        label_input = tf.reshape(label_input, [-1, 7,7,1])

        # get latent vec input
        latent_input = self.transform_latent_vecs(latent_vecs) # [7,7,128]

        # concat with latent dim with projected label (as an additional channel => 129 total channels)
        generator_input = tf.concat([latent_input, label_input], axis=3) # channel axis = 3
        # model forward pass (same as unconditional conv2D)
        return self.model(generator_input)

# def generator():
#     dense_size = 128 * 7 * 7 #we want 128 channels of size 7x7 => project latent vect
#     model = Sequential([
#         # project for upsampling
#         Dense(dense_size),
#         LeakyReLU(alpha=0.2),
#         Reshape([7,7,128]),
#         # upsample (14x14)
#         Conv2DTranspose(filters=128, kernel_size=(4,4), strides=(2,2), padding='same'),
#         LeakyReLU(alpha=0.2),
#         # upsample (28x28)
#         Conv2DTranspose(filters=128, kernel_size=(4,4), strides=(2,2), padding='same'),
#         LeakyReLU(alpha=0.2),
#         # generate single-channel 28x28 img 
#         Conv2D(1, (7,7), activation='tanh', padding='same')
#     ])
#     return model
