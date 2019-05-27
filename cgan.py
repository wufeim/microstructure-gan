import os
import sys

import numpy as np
import pandas as pd

import cv2

import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Reshape
from keras.layers import Conv2D, Conv2DTranspose, UpSampling2D
from keras.layers import LeakyReLU, Dropout, BatchNormalization
from keras.layers.merge import concatenate
from keras.optimizers import RMSprop, Adam

class CGAN(object):

    def __init__(self, img_rows=28, img_cols=28, channels=1):
        self.model_name = 'cgan_microstructure'
        self.num_classes = 5
        self.latent_size = 100
        self.batch_size = 20
        self.train_steps = 30000
        self.lr = 5e-6
        self.decay = 6e-8

        self.img_rows = img_rows
        self.img_cols = img_cols
        self.channels = channels
        self.discriminator = None
        self.generator = None
        self.adversarial_model = None
        self.discriminator_model = None

    def build_discriminator(self, input_layer, label_layer):
        if self.discriminator:
            return self.discriminator
        m = Dense(self.img_rows, self.img_cols)(label_layer)
        m = Reshape(self.img_rows, self.img_cols, 1)(x)
        m = concatenate([input_layer, x])

        m = ZeroPadding2D((1,1))(m)
        m = Conv2D(64, 3, 3, activation='relu')(m)
        m = ZeroPadding2D((1,1))
        m = Conv2D(64, 3, 3, activation='relu')(m)
        m = MaxPooling2D((2,2), strides=(2,2))(m)

        m = ZeroPadding2D((1,1))(m)
        m = Conv2D(128, 3, 3, activation='relu')(m)
        m = ZeroPadding2D((1,1))(m)
        m = Conv2D(128, 3, 3, activation='relu')(m)
        m = MaxPooling2D((2,2), strides=(2,2))(m)

        m = ZeroPadding2D((1,1))(m)
        m = Conv2D(256, 3, 3, activation='relu')(m)
        m = ZeroPadding2D((1,1))(m)
        m = Conv2D(256, 3, 3, activation='relu')(m)
        m = ZeroPadding2D((1,1))(m)
        m = Conv2D(256, 3, 3, activation='relu')(m)
        m = ZeroPadding2D((1,1))(m)
        m = Conv2D(256, 3, 3, activation='relu')(m)
        m = MaxPooling2D((2,2), strides=(2,2))(m)

        m = ZeroPadding((1,1))(m)
        m = Conv2D(512, 3, 3, activation='relu')(m)
        m = ZeroPadding((1,1))(m)
        m = Conv2D(512, 3, 3, activation='relu')(m)
        m = ZeroPadding((1,1))(m)
        m = Conv2D(512, 3, 3, activation='relu')(m)
        m = ZeroPadding((1,1))(m)
        m = Conv2D(512, 3, 3, activation='relu')(m)
        m = MaxPooling2D((2,2), strides=(2,2))(m)

        m = ZeroPadding((1,1))(m)
        m = Conv2D(512, 3, 3, activation='relu')(m)
        m = ZeroPadding((1,1))(m)
        m = Conv2D(512, 3, 3, activation='relu')(m)
        m = ZeroPadding((1,1))(m)
        m = Conv2D(512, 3, 3, activation='relu')(m)
        m = ZeroPadding((1,1))(m)
        m = Conv2D(512, 3, 3, activation='relu')(m)
        m = MaxPooling2D((2,2), strides=(2,2))(m)
        m = Flatten()(m)
        m = Dense(256, activation='relu')(m)
        m = Dense(1, activation='relu')(m)

        self.discriminator = Model([input_layer, label_layer], m, name='discriminator')
        return self.discriminator

    def build_generator(self, input_layer, label_layer):
        if self.generator:
            return self.generator
        img_x = self.img_rows // 4
        img_y = self.img_cols // 4
        kernel_size = 5
        layer_filters = [128, 64, 32, 1]
        x = concatenate([input_layer, label_layer], axis=1)
        x = Dense(img_x * img_y * layer_filters[0])(x)
        x = Reshape(img_x, img_y, layer_filters[0])(x)
        for filters in layer_filters:
            if filters > layer_filters[-2]:
                strides = 2
            else:
                strides = 1
            x = BatchNormalization()(x)
            x = Activation('relu')(x)
            x = Conv2DTranspose(filters=filters,
                                kernel_size=kernel_size,
                                strides=strides,
                                padding='same')(x)
        x = Activation('sigmoid')(x)
        self.generator = Model([input_layer, label_layer], x, name='generator')
        return self.generator

    def train(self):
        generator = self.generator
        discriminator = self.discriminator
        adversarial = self.adversarial_model
        batch_size = self.batch_size
        latent_size = self.latent_size
        num_classes = self.num_classes
        train_steps = self.train_steps

        # load images
        x_train, y_train

        save_interval = 500
        noise_input = np.random.uniform(-1.0, 1.0, size=[16, latent_size])
        noise_class = np.eye(num_classes)[np.arange(0, 16) % num_classes]
        train_size = x_train.shape[0]

        print(model_name,
              "Labels for generated images: ",
              np.argmax(noise_class, axis=1))
        
        for i in range(train_steps):
            rand_indexes = np.random.randint(0, train_size, size=batch_size)
            real_images = x_train[rand_indexes]
            real_labels = y_train[rand_indexes]
            noise = np.random.uniform(-1.0, 1.0, size=[batch_size, latent_size])
            fake_labels = np.eye(num_classes)[np.random.choice(num_classes, batch_size)]

            fake_images = generator.predict([noise, fake_labels])
            x = np.concatenate((real_images, fake_images))
            labels = np.concatenate((real_labels, fake_labels))

            y = np.ones([2 * batch_size, 1])
            y[batch_size:, :] = 0.0
            loss, acc = discriminator.train_on_batch([x, labels], y)
            log = "%d: [discriminator loss: %f, acc: %f]" % (i, loss, acc)

            # traint the adversarial_model
            noise = np.random.uniform(-1.0, 1.0, size=[batch_size, latent_size])
            fake_labels = np.eye(num_classes)[np.random.choice(num_classes, batch_size)]

            fake_images = generator.predict([noise, fake_labels])
            x = np.concatenate((real_images, fake_images))
            labels = np.concatenate((real_labels, fake_labels))

            y = np.ones([2 * batch_size, 1])
            y[batch_size:, :] = 0.0
            loss, acc = discriminator.train_on_batch([x, labels], y)
            log = "%d: [discriminator loss: %f, acc: %f]" % (i, loss, acc)

            noise = np.random.uniform(-1.0, 1.0, size=[batch_size, latent_size])
            fake_labels = np.eye(num_classes)[np.random.choice(num_classes, batch_size)]
            y = np.ones([batch_size, 1])
            loss, acc = adversarial.train_on_batch([noise, fake_labels], y)
            log = "%s [adversarial loss: %f, acc: %f]" % (log, loss, acc)
            print(log)
            if (i + 1) % save_interval == 0:
                if (i + 1) == train_steps:
                    show = True
                else:
                    show = False

                plot_images(generator,
                            noise_input=noise_input,
                            noise_class=noise_class,
                            show=show,
                            step=(i + 1),
                            model_name=model_name)
        generator.save(model_name + ".h5")

    def plot_images(generator, noise_input, noise_class, show=False, step=0, model_name='cgan'):
        os.makedirs(model_name, exist_ok=True)
        filename = os.path.join(model_name, "%05d.png" % step)
        images = generator.predict([noise_input, noise_class])
        print(model_name , " labels for generated images: ", np.argmax(noise_class, axis=1))
        plt.figure(figsize=(2.2, 2.2))
        num_images = images.shape[0]
        image_size = images.shape[1]
        rows = int(math.sqrt(noise_input.shape[0]))
        for i in range(num_images):
            plt.subplot(rows, rows, i + 1)
            image = np.reshape(images[i], [image_size, image_size])
            plt.imshow(image, cmap='gray')
            plt.axis('off')
        plt.savefig(filename)
        if show:
            plt.show()
        else:
            plt.close('all')

    def build_and_train_models(self):
        img_rows = self.img_rows
        img_cols = self.img_cols
        num_classes = self.num_classes
        model_name = self.model_name
        latent_size = self.latent_size
        batch_size = self.batch_size
        train_steps = self.train_steps
        lr = self.lr
        decay = self.decay
        input_shape = (img_rows, img_cols, 1)
        label_shape = (num_classes, )

        input_layer = Input(shape=input_shape, name='discriminator_input')
        label_layer = Input(shape=label_shape, name='class_labels')

        self.discriminator_model = build_discriminator(input_layer, label_layer)
        optimizer = Adam(lr=lr)
        self.discriminator_model.compile(loss='binary_crossentropy',
                                         optimizer=optimizer,
                                         metrics=['acc'])
        self.discriminator_model.summary()

        input_shape = (latent_size, )
        input_layer = Input(shape=input_shape, name='z_input')
        generator = build_generator(input_layer, label_layer)
        generator.summary()

        optimizer = RMSprop(lr=lr*0.5, decay=decay*0.5)
        self.discriminator_model.trainable = False
        output_layer = discriminator([generator([input_layer, label_layer]), label_layer])
        self.adversarial_model = Model([input_layer, label_layer],
                                       output_layer,
                                       name=model_name)
        self.adversarial_model.compile(loss='binary_crossentropy',
                                       optimizer=optimizer,
                                       metrics=['acc'])
        self.adversarial_model.summary()

if __name__=='__main__':
    cgan = CGAN()
    cgan.build_and_train_models()
        
