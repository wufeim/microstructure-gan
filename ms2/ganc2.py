import os
import sys
import glob
import time

import cv2

import numpy as np

from keras.layers import Input, Dense, Reshape, Flatten, Dropout, multiply
from keras.layers import BatchNormalization, Activation, Embedding, ZeroPadding2D
from keras.layers import Concatenate
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D, Conv2DTranspose
from keras.models import Sequential, Model
from keras.optimizers import Adam
from keras.applications import VGG19

import matplotlib.pyplot as plt


class GAN():


    def __init__(self):

        self.img_rows = 960
        self.img_cols = 960
        self.channels = 1
        self.img_shape = (self.img_rows, self.img_cols, self.channels)

        self.latent_dim = 128
        self.ratio = 32

        d_optimizer = Adam(lr=1e-3, beta_1=0.5)
        g_optimizer = Adam(lr=1e-3, beta_1=0.5)

        self.class_names = ['DUM562']

        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='binary_crossentropy',
                                   optimizer=g_optimizer,
                                   metrics=['acc'])
        
        self.generator = self.build_generator()
        noise = Input(shape=(self.latent_dim, ))
        img = self.generator(noise)
        
        self.discriminator.trainable = False

        validity = self.discriminator(img)

        self.combined = Model(noise, validity)
        self.combined.compile(loss='binary_crossentropy',
                              optimizer=d_optimizer)
        
        names = []
        for c in self.class_names:
            names += glob.glob('data/{:s}/*.tif'.format(c))
        self._img_names = np.array(names)

        names = []
        for c in self.class_names:
            names += glob.glob('data/{:s}/*.tif'.format(c))
        self._img_names = np.array(names)


    def build_discriminator(self):
        
        model = Sequential()

        model.add(Conv2D(32, (4, 4), strides=2, padding='same', activation='relu'))
        model.add(BatchNormalization())
        model.add(Conv2D(64, (4, 4), strides=2, padding='same', activation='relu'))
        model.add(BatchNormalization())
        model.add(Conv2D(128, (4, 4), strides=2, padding='same', activation='relu'))
        model.add(BatchNormalization())
        model.add(Conv2D(256, (4, 4), strides=2, padding='same', activation='relu'))
        model.add(BatchNormalization())
        model.add(Flatten())
        model.add(Dense(512, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(1, activation='sigmoid'))

        img = Input(shape=self.img_shape)
        validity = model(img)

        m = Model(img, validity, name='Discriminator')

        print('\nDiscriminator:')
        print(m.summary())

        return m


    def build_generator(self):
        
        resized_size = (self.img_rows // self.ratio, self.img_cols // self.ratio, self.channels)
        
        model = Sequential()

        model.add(Dense(np.prod(np.array(resized_size)), activation='relu'))
        model.add(Reshape(resized_size))
        model.add(BatchNormalization())
        model.add(Conv2DTranspose(256, (4, 4), strides=2, padding='same', activation='relu'))
        model.add(BatchNormalization())
        model.add(Conv2DTranspose(128, (4, 4), strides=2, padding='same', activation='relu'))
        model.add(BatchNormalization())
        model.add(Conv2DTranspose(64, (4, 4), strides=2, padding='same', activation='relu'))
        model.add(BatchNormalization())
        model.add(Conv2DTranspose(32, (4, 4), strides=2, padding='same', activation='relu'))
        model.add(BatchNormalization())
        model.add(Conv2DTranspose(1, (4, 4), strides=2, padding='same', activation='tanh'))

        noise = Input(shape=(self.latent_dim, ))
        img = model(noise)

        m = Model(noise, img, name='generator')
        
        print('\nGenerator:')
        print(m.summary())

        return m


    def load_data(self, n):
        
        idx = np.random.randint(0, len(self._img_names), n)

        imgs = None
        for x in self._img_names[idx]:
            img = cv2.imread(x)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = img.astype('float32') / 255
            img = img[:960, :960]
            img = np.expand_dims(img, axis=-1)
            img = np.expand_dims(img, axis=0)
            if imgs is not None:
                imgs = np.vstack((imgs, img))
            else:
                imgs = img

        return imgs


    def sample_images(self, epoch):
        
        r, c = 3, 3
        noise = np.random.normal(0, 1, (r * c, self.latent_dim))
        gen_imgs = self.generator.predict(noise)
        print(gen_imgs.shape)

        fig, ax = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                ax[i, j].imshow(gen_imgs[cnt, :, :, 0], cmap='gray')
                ax[i, j].axis('off')
                cnt += 1
        fig.savefig('ms-gan-2/epoch-{:d}.png'.format(epoch), dpi=800)
        plt.close()


    def train(self, epochs, batch_size=128, sample_interval=50):
        
        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))

        for epoch in range(1, epochs+1):
            
            # ---------------------
            #  Train Discriminator
            # ---------------------

            imgs = self.load_data(batch_size)

            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
            gen_imgs = self.generator.predict(noise)

            d_loss_real = self.discriminator.train_on_batch(imgs, valid)
            d_loss_fake = self.discriminator.train_on_batch(imgs, fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # -----------------
            #  Train Generator
            # -----------------

            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))

            g_loss = self.combined.train_on_batch(noise, valid)

            print('Epoch {:3d}: [D loss: {:f}, acc: {:.2f}%] [G loss: {:f}]'.format(epoch, d_loss[0], d_loss[1]*100, g_loss))

            if epoch % sample_interval == 0:
                self.sample_images(epoch)


if __name__=='__main__':

    start_time = time.time()

    os.makedirs('ms-gan-2', exist_ok=True)
    gan = GAN()
    gan.train(epochs=2000, batch_size=32, sample_interval=50)

    end_time = time.time()

    print('\nDone. Time elapsed: {:.0f} secs.'.format(end_time-start_time))

