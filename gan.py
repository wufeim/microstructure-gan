import os
import sys

import numpy as np
import pandas as pd

import cv2

from matplotlib import pyplot as plt

from keras import optimizers

class GAN():

    def __init__(self):
        self.channels = 1

        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='binary_crossentropy',
                                   optimizer=optimizers.Adam(lr=5e-6),
                                   metrics=['acc'])
        self.generator = self.build_generator()

        # create noises and generate images
        z = Input(shape=(self.latent_dim, ))
        img = self.generator(z)

        self.discriminator.trainable = False
        
        validality = self.discriminator(img)

        self.gan = Model(z, validality)
        self.gan.compile(loss='binary_crossentropy',
                         optimizer=optimizers.Adam(lr=2e-5))

    def build_discriminator():

    def build_generator():

