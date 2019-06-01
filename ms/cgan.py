import os
import sys
import glob

import numpy as np

from keras.layers import Input, Dense, Reshape, Flatten, Dropout, multiply
from keras.layers import BatchNormalization, Activation, Embedding, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D, Conv2DTranspose
from keras.models import Sequential, Model
from keras.optimizers import Adam

from keras_contrib import InstanceNormalization

import matplotlib.pyplot as plt

class CGAN():

    def __init__(self):

        self.img_rows = 960
        self.img_cols = 1280
        self.channels = 1
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.num_classes = 5
        self.class_names = ['DUM555', 'DUM560', 'DUM562', 'DUM587', 'DUM588']
        self.latent_dim = 1000

        optimizer = Adam(0.0002, 0.5)

        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='binary_crossentropy',
                                   optimizer=optimizer,
                                   metrics=['accuracy'])

        self.generator = self.build_generator()

        noise = Input(shape=(self.latent_dim, ))
        label = Input(shape=(1, ))
        img = self.generator([noise, label])

        self.discriminator.trainable = False

        valid = self.discriminator([img, label])

        self.combined = Model([noise, label], valid)
        self.combined.compile(loss=['binary_crossentropy'],
                              optimizer=optimizer)

        # initialization for training images
        # directory layout:
        #     .
        #     \-- data/
        #         |-- class1/
        #         |   |-- *.tif
        #         |   \-- *.tif
        #         \-- class2/
        #             |-- *.tif
        #             \-- *.tif
        self._img_names = []
        for c in self.class_names:
            self._img_names += glob.glob('data/{:s}/*.tif'.format(c))

    def build_discriminator(self):

        model = Sequential()

        # discriminator structure from pix2pixHD
        model.add(Conv2D(64, (4, 4), strides=2))
        model.add(LeakyReLU(alpha=0.2))

        model.add(Conv2D(128, (4, 4), strides=2))
        model.add(InstanceNormalization())
        model.add(LeakyReLU(alpha=0.2))

        model.add(Conv2D(256, (4, 4), strides=2))
        model.add(InstanceNormalization())
        model.add(LeakyReLU(alpha=0.2))

        model.add(Conv2D(512, (4, 4), strides=2))
        model.add(InstanceNormalization())
        model.add(LeakyReLU(alpha=0.2))

        model.add(Flatten())
        model.add(Dense(1, activation='sigmoid'))

        img = Input(shape=self.img_shape)
        flat_img = Flatten()(img)

        label = Input(shape=(1, ), dtype='int32')
        label_embedding = Flatten()(Embedding(self.num_classes, np.prod(self.img_shape))(label))

        model_input = multiply([flat_img, label_embedding])
        model_input = Reshape(self.img_shape)(model_input)

        validity = model(model_input)

        m = Model([img, label], validity)

        print('\n\nDiscriminator:')
        m.summary()

        return m


    def build_generator(self):
    
        model = Sequential()

        model.add(Dense((self.img_shape[0] // 32) * (self.img_shape[1] // 32), input_dim=self.latent_dim))
        model.add(Dense((self.img_shape[0] // 16) * (self.img_shape[1] // 16)))
        model.add(Reshape((self.img_shape[0] // 16, self.img_shape[1] // 16, 1)))
        model.add(BatchNormalization())
        model.add(Conv2DTranspose(512, (4, 4), strides=2, padding='same', activation='relu'))
        model.add(BatchNormalization())
        model.add(Conv2DTranspose(256, (4, 4), strides=2, padding='same', activation='relu'))
        model.add(BatchNormalization())
        model.add(Conv2DTranspose(128, (4, 4), strides=2, padding='same', activation='relu'))
        model.add(BatchNormalization())
        model.add(Conv2DTranspose(64, (4, 4), strides=2, padding='same', activation='relu'))
        model.add(BatchNormalization())
        model.add(Conv2DTranspose(1, (4, 4), strides=1, padding='same', activation='tanh'))

        noise = Input(shape=(self.latent_dim, ))

        label = Input(shape=(1, ), dtype='int32')
        label_embedding = Flatten()(Embedding(self.num_classes, self.latent_dim)(label))

        model_input = multiply([noise, label_embedding])

        img = model(model_input)

        m = Model([noise, label], img)

        print('\n\nGenerator:')
        m.summary()

        return m


    def train(self, epochs, batch_size, sample_interval):

        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))

        for epoch in range(1, epochs+1):
            
            # ---------------------
            #  Train Discriminator
            # ---------------------

            (imgs, labels) = self.load_imgs_and_labels(n=batch_size)

            noise = np.random.normal(0, 1, (batch_size, 100))

            gen_imgs = self.generator.predict([noise, labels])

            d_loss_real = self.discriminator.train_on_batch([imgs, labels], valid)
            d_loss_fake = self.discriminator.train_on_batch([gen_imgs, labels], fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # -----------------
            #  Train Generator
            # -----------------

            sampled_labels = np.random.randint(0, 10, batch_size).reshape(-1, 1)

            g_loss = self.combined.train_on_batch([noise, sampled_labels], valid)

            print('Epoch: {:d} [D loss: {:f}, acc: {:.2f}%] [G loss: {:f}]'.format(epoch, d_loss[0], 100*d_loss[1], g_loss))

            if epoch % sample_interval == 0:
                self.sample_images(epoch)

    def load_imgs_and_labels(self, n):
        
        idx = np.random.randint(0, len(self._img_names), n)

        imgs = None
        for x in self._img_names[idx]:
            img = cv2.imread(x)
            if imgs:
                imgs = np.vstack((imgs, img))
            else:
                imgs = np.array([img])
        
        labels = [x.split('/')[1] for x in self._img_names[idx]]
        labels = np.array([self.class_names.index(x) for x in labels])

        return (imgs, labels)

    def sample_images(self, epoch):

        r, c = 5, 3
        noise = np.random.normal(0, 1, (r * c, 100))
        sampled_labels = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4])

        gen_imgs = self.generator.predict([noise, sampled_labels])

        gen_imgs = 0.5 * gen_imgs + 0.5

        fig, ax = plt.subplots(r, c)
        cnt = 0
        for j in range(c):
            for i in range(r):
                ax[i, j].imshow(gen_imgs[cnt, :, :, 0], cmap='gray')
                ax[i, j].set_title(self.class_names[sampled_labels[cnt]])
                ax[i, j].axis('off')
                cnt += 1
        fig.savefig('gen-ms/{:d}.png'.format(epoch))
        plt.close()

if __name__=='__main__':

    os.makedirs('gen-ms', exist_ok=True)
    cgan = CGAN()
    cgan.train(epochs=1, batch_size=32, sample_interval=500)

