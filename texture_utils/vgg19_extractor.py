import os
import tensorflow as tf
from collections import OrderedDict

import numpy as np
import time
import inspect

from .vgg19_trainable import Vgg19, VGG_MEAN

VGG_MEAN_RGB = list(reversed(VGG_MEAN))

class Vgg19Extractor(Vgg19):
    def __init__(self, vgg19_npy_path=None, trainable=False,
            is_rgb_input=True, use_avg_pool=True, name="vgg19_extractor"):
        if vgg19_npy_path is None:
            path = inspect.getfile(Vgg19Extractor)
            path = os.path.abspath(os.path.join(path, os.pardir))
            vgg19_npy_path = os.path.join(path, "vgg19_normalised_rgb.npz")
            
        Vgg19.__init__(self, vgg19_npy_path, trainable)
        if is_rgb_input:
            w_conv1_1 = self.data_dict['conv1_1'][0]
            assert w_conv1_1.shape == (3, 3, 3, 64)
            #self.data_dict['conv1_1'][0] = w_conv1_1[:,:,::-1,:]
        self.is_rgb_input = is_rgb_input
        self.use_avg_pool = use_avg_pool
        self.name = name
        print("npy file loaded from", vgg19_npy_path)

    def build(self, input_image, name="vgg19"):
        """Load variable from npy to build the VGG19 feature extractor

        Parameters
        ----------
        input_image : tf.Tensor
            input image [batch, height, width, 3] values scaled [0, 1]
        """

        start_time = time.time()
        print("build VGG19 model started")

        with tf.name_scope(name):
            input_image_scaled = input_image * 255.0

            if self.is_rgb_input:
                input_image = input_image_scaled - VGG_MEAN_RGB
            else:
                input_image = input_image_scaled - VGG_MEAN
            
            if self.use_avg_pool:
                pool = self.avg_pool
            else:
                pool = self.max_pool

            self.conv1_1 = self.conv_layer(input_image, 3, 64, "conv1_1")
            self.conv1_2 = self.conv_layer(self.conv1_1, 64, 64, "conv1_2")
            self.pool1 = pool(self.conv1_2, "pool1")

            self.conv2_1 = self.conv_layer(self.pool1, 64, 128, "conv2_1")
            self.conv2_2 = self.conv_layer(self.conv2_1, 128, 128, "conv2_2")
            self.pool2 = pool(self.conv2_2, "pool2")

            self.conv3_1 = self.conv_layer(self.pool2, 128, 256, "conv3_1")
            self.conv3_2 = self.conv_layer(self.conv3_1, 256, 256, "conv3_2")
            self.conv3_3 = self.conv_layer(self.conv3_2, 256, 256, "conv3_3")
            self.conv3_4 = self.conv_layer(self.conv3_3, 256, 256, "conv3_4")
            self.pool3 = pool(self.conv3_4, "pool3")

            self.conv4_1 = self.conv_layer(self.pool3, 256, 512, "conv4_1")
            self.conv4_2 = self.conv_layer(self.conv4_1, 512, 512, "conv4_2")
            self.conv4_3 = self.conv_layer(self.conv4_2, 512, 512, "conv4_3")
            self.conv4_4 = self.conv_layer(self.conv4_3, 512, 512, "conv4_4")
            self.pool4 = pool(self.conv4_4, "pool4")

            # Explicitly close the NpzFile object to avoid leaking file descriptors
            self.data_dict.close()
            #self.output_names = ["conv1_1", "pool1", "pool2", "pool3", "pool4"]
            #self.outputs = [self.conv1_1, self.pool1, self.pool2, self.pool3, self.pool4]
        print(("build model finished: %ds" % (time.time() - start_time)))
        return OrderedDict([
            ("conv1_1", self.conv1_1),
            ("pool1", self.pool1),
            ("pool2", self.pool2),
            ("pool3", self.pool3),
            ("pool4", self.pool4)
        ])