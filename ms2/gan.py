import os
import sys
import glob
import time

import cv2

import numpy as np 
import tensorflow as tf

from keras.layers import Input, Dense, Reshape, Flatten, Dropout, multiply
from keras.layers import BatchNormalization, Activation, Embedding, ZeroPadding2D
from keras.layers import Concatenate, Lambda
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D, Conv2DTranspose
from keras.models import Sequential, Model
from keras.optimizers import Adam
from keras.applications import VGG19

import matplotlib.pyplot as plt


def build_discriminator(input_shape, vgg_input_shape=(256, 256, 3)):

    conv_base = VGG19(weights='imagenet', include_top=False, input_shape=vgg_input_shape)
    set_trainable = False
    for layer in conv_base.layers:
        if layer.name == 'block5_conv1':
            set_trainable = True
        if set_trainable:
            layer.trainable = True
        else:
            layer.trainable = False
    m = models.Sequential()
    m.add(Input(shape=input_shape))
    m.add(Lambda(lambda image: tf.image.resize_images(image, vgg_input_shape)))
    m.add(conv_base)
    m.add(Flatten())
    m.add(Dense(256, activation='relu'))
    m.add(Dense(5, activation='softmax'))
    return m


if __name__=='__main__':

    img_rows = 1024
    img_cols = 1024
    channels = 3
    d_optimizer = Adam(lr=5e-6)

    img_shape = (img_rows, img_cols, channels)
    
    discriminator = build_discriminator(img_shape)
    discriminator.compile(loss='binary_crossentropy', optimizer=d_optimizer, metircs=['acc'])

    # Train the discriminator

