import os
import tensorflow as tf
from collections import OrderedDict

import numpy as np
from tictoc import Timer
import inspect

VGG_MEAN = [103.939, 116.779, 123.68] # BGR [0, 255]
VGG_MEAN_RGB = list(reversed(VGG_MEAN))
PARAM_SHAPE = OrderedDict([
    ("conv1_1", [(3, 3, 3 ,64), (64,)]),
    ("conv1_2", [(3, 3, 64, 64), (64,)]),
    ("pool1", []),
    #
    ("conv2_1", [(3, 3, 64, 128), (128,)]),
    ("conv2_2", [(3, 3, 128, 128), (128,)]),
    ("pool2", []),
    #
    ("conv3_1", [(3, 3, 128, 256), (256,)]),
    ("conv3_2", [(3, 3, 256, 256), (256,)]),
    ("conv3_3", [(3, 3, 256, 256), (256,)]),
    ("conv3_4", [(3, 3, 256, 256), (256,)]),
    ("pool3", []),
    #
    ("conv4_1", [(3, 3, 256, 512), (512,)]),
    ("conv4_2", [(3, 3, 512, 512), (512,)]),
    ("conv4_3", [(3, 3, 512, 512), (512,)]),
    ("conv4_4", [(3, 3, 512, 512), (512,)]),
    ("pool4", []),
    #
    ("conv5_1", [(3, 3, 512, 512), (512,)]),
    ("conv5_2", [(3, 3, 512, 512), (512,)]),
    ("conv5_3", [(3, 3, 512, 512), (512,)]),
    ("conv5_4", [(3, 3, 512, 512), (512,)]),
    ("pool5", []),
    #
    ("fc6", [(7*7*512, 4096), (4096,)]),
    ("fc7", [(4096, 4096), (4096,)]),
    ("fc8", [(4096, 1000), (1000,)])
])
RECP_FIELD = OrderedDict([
    ("conv1_1", 3),
    ("conv1_2", 5),
    ("pool1", 6),
    ("conv2_1", 10),
    ("conv2_2", 14),
    ("pool2", 16),
    ("conv3_1", 24),
    ("conv3_2", 32),
    ("conv3_3", 40),
    ("conv3_4", 48),
    ("pool3", 52),
    ("conv4_1", 68),
    ("conv4_2", 84),
    ("conv4_3", 100),
    ("conv4_4", 116),
    ("pool4", 124)
])
ACCM_STRIDE = OrderedDict([
    ("conv1_1", 1),
    ("conv1_2", 1),
    ("pool1", 2),
    ("conv2_1", 2),
    ("conv2_2", 2),
    ("pool2", 4),
    ("conv3_1", 4),
    ("conv3_2", 4),
    ("conv3_3", 4),
    ("conv3_4", 4),
    ("pool3", 8),
    ("conv4_1", 8),
    ("conv4_2", 8),
    ("conv4_3", 8),
    ("conv4_4", 8),
    ("pool4", 16)
])

def get_default_vgg19_path():
    path = inspect.getfile(get_default_vgg19_path)
    path = os.path.abspath(os.path.join(path, os.pardir))
    path = os.path.join(path, "vgg19_normalised.npz")
    if os.path.exists(path):
        print("VGG19 path is found at", path)
        return path
    else:
        print("VGG19 path is not found")
        return None

def avg_pool(bottom, name):
    return tf.nn.avg_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME", name=name)

def max_pool(bottom, name):
    return tf.nn.max_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME", name=name)

class Vgg19:
    def __init__(self, vgg19_path=None, trainable=False, padding="SAME",
            is_rgb_input=True, use_avg_pool=True, topmost="fc8", name="vgg19"):
        if vgg19_path is None:
            vgg19_path = get_default_vgg19_path()
        if vgg19_path.endswith(".npy"):
            data_dict = np.load(vgg19_path, encoding='latin1').item()
        elif vgg19_path.endswith(".npz"):
            data_dict = np.load(vgg19_path, encoding='latin1', allow_pickle=True)
        else:
            raise ValueError("vgg19_npy_path should be a 'npy' or 'npz' file")
        
        self.var_dict = {}
        self.trainable = trainable
        assert padding in ["SAME", "VALID"]
        self.padding = padding
        self.is_rgb_input = is_rgb_input
        self.use_avg_pool = use_avg_pool
        self.topmost = topmost
        self.name = name

        # load parameters
        self.data_dict = dict()
        for layer in PARAM_SHAPE:
            if layer in data_dict:
                self.data_dict[layer] = data_dict[layer]
            if layer == topmost:
                break
        if is_rgb_input:
            w_conv1_1 = self.data_dict["conv1_1"][0]
            assert w_conv1_1.shape == PARAM_SHAPE["conv1_1"][0]
            self.data_dict['conv1_1'][0] = w_conv1_1[:,:,::-1,:]
        # Explicitly close the NpzFile object to avoid leaking file descriptors
        data_dict.close()
        print("Loaded vgg19 from", vgg19_path)

    def build(self, input_image, topmost=None, padding=None, name="vgg19"):
        """Load variable from npy to build the VGG19 feature extractor

        Parameters
        ----------
        input_image : tf.Tensor
            input image [batch, height, width, 3] values scaled [0, 1]
        """
        layers = OrderedDict()
        with tf.name_scope(name):
            input_image_scaled = input_image * 255.
            if self.is_rgb_input:
                input_image = input_image_scaled - VGG_MEAN_RGB
            else:
                input_image = input_image_scaled - VGG_MEAN

            if topmost is None:
                topmost = self.topmost
            if padding is None:
                padding = self.padding
            if self.use_avg_pool:
                pool = avg_pool
            else:
                pool = max_pool

            x = input_image
            for layer in PARAM_SHAPE:
                if layer.startswith("conv"):
                    x = self.conv_layer(x, padding, layer)
                elif layer.startswith("pool"):
                    x = pool(x, layer)
                elif layer.startswith("fc"):
                    x = self.fc_layer(x, layer, layer != "fc8")
                layers[layer] = x
                if layer == topmost:
                    break
            print("Build vgg19 model:", name)
        return layers
        
    def conv_layer(self, bottom, padding, name):
        with tf.variable_scope(name):
            filt = self.get_var(name, 0)
            bias = self.get_var(name, 1)
            #print(name, bottom.shape, filt.shape, bias.shape)
            x = tf.nn.conv2d(bottom, filt, [1, 1, 1, 1], padding=padding)
            x = tf.nn.bias_add(x, bias)
            x = tf.nn.relu(x)
            return x
    
    def fc_layer(self, bottom, name, relu=True):
        with tf.variable_scope(name):
            weight = self.get_var(name, 0)
            bias = self.get_var(name, 1)
            x = tf.reshape(bottom, [-1, PARAM_SHAPE[name][0][0]])
            x = tf.matmul(x, weight)
            x = tf.nn.bias_add(x, bias)
            if relu:
                x = tf.nn.relu(bias)
            return x

    def get_var(self, name, idx, stddev=0.001):
        var = self.var_dict.get((name, idx))
        if var is None:
            if self.data_dict is not None and name in self.data_dict:
                value = self.data_dict[name][idx]
            else:
                value = tf.truncated_normal(PARAM_SHAPE[name][idx], 0., stddev)
            var_name = "%s_%d" % (name, idx)
            if self.trainable:
                var = tf.Variable(value, name=var_name)
            else:
                var = tf.constant(value, dtype=tf.float32, name=var_name)
            
            self.var_dict[(name, idx)] = var
        return var

    def save_npy(self, sess, npy_path="vgg19-save.npy"):
        assert isinstance(sess, tf.Session)

        data_dict = dict()

        for (name, idx), var in list(self.var_dict.items()):
            var_out = sess.run(var)
            if name not in data_dict:
                data_dict[name] = dict()
            data_dict[name][idx] = var_out

        np.save(npy_path, data_dict)
        print("Save param to", npy_path)
        return npy_path

    def get_var_count(self):
        count = 0
        for (name, idx) in list(self.var_dict.keys()):
            count += np.prod(PARAM_SHAPE[name][idx])
        return count