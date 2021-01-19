import numpy as np
from tensorpack.compat import tfv1 as tf

from tensorpack.tfutils.common import get_tf_version_tuple
from tensorpack.utils.argtools import get_data_format, shape2d, shape4d, log_once
from tensorpack.models.common import VariableHolder, layer_register
from tensorpack.models.tflayer import convert_to_tflayer_args, rename_get_variable

def get_weight(shape, gain=1, use_wscale=True, lrmul=1, adaptive_lr=True, name='W'):
    assert len(shape) == 4 or len(shape) == 2
    fan_in = np.prod(shape[:-1]) # [ksize, ksize, cin, cout] or [cin, cout]
    he_std = gain / np.sqrt(fan_in) # He init

    if lrmul == 0:
        init = tf.initializers.random_normal(0, he_std)
        return tf.get_variable(name, shape=shape, initializer=init, trainable=False)

    # Equalized learning rate and custom learning rate multiplier.
    if not adaptive_lr:
        lrmul = np.sqrt(lrmul)
    if use_wscale:
        init_std = 1. / lrmul
        runtime_coef = he_std * lrmul
    else:
        init_std = he_std / lrmul
        runtime_coef = lrmul

    # Create variable
    init = tf.initializers.random_normal(0, init_std)
    return tf.get_variable(name, shape=shape, initializer=init) * runtime_coef

def get_bias(chan, base_std=0, use_wscale=True, lrmul=1, adaptive_lr=True, name='b'):
    if lrmul == 0:
        init = tf.initializers.random_normal(0, base_std)
        return tf.get_variable(name, shape=[chan], initializer=init, trainable=False)

    if not adaptive_lr:
        lrmul = np.sqrt(lrmul)
    if use_wscale and base_std != 0:
        init_std = 1. / lrmul
        runtime_coef = base_std * lrmul
        trainable = True
    else:
        init_std = base_std / lrmul
        runtime_coef = lrmul

    init = tf.initializers.random_normal(0, init_std)
    return tf.get_variable(name, shape=[chan], initializer=init) * runtime_coef

def dense(x, fmaps, gain=1, use_wscale=True, lrmul=1, name='dense'):
    shape = x.get_shape().as_list()
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        if len(shape) > 2:
            cin = np.prod(shape[1:])
            x = tf.reshape(x, [-1, cin])
        else:
            assert len(shape) == 2
            cin = shape[1]
        w = get_weight([cin, fmaps], gain=gain, use_wscale=use_wscale, lrmul=lrmul, name='W')
        return tf.matmul(x, tf.cast(w, x.dtype)), w

def mod_conv2d(x, y, fmaps, kernel, demodulate=True, gain=1, use_wscale=True, lrmul=1,
        fused_modconv=True, eps=1e-8, padding='SAME', name="mod_conv2d"):
    shape = x.get_shape().as_list() # [n, h, w, c]
    cin = shape[-1]
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        # Get weight
        w = get_weight([kernel, kernel, cin, fmaps], gain=gain, use_wscale=use_wscale, lrmul=lrmul, name='W')
        ww = w[tf.newaxis] # introduce minibatch dimension

        # Modulate
        s = get_bias(cin, base_std=0, use_wscale=use_wscale, lrmul=lrmul, name='bs') + 1
        vh = VariableHolder(W=w, bs=s)
        s = tf.tile(s[tf.newaxis], [tf.shape(x)[0], 1]) # introduce minibatch dimension
        if y is not None:
            y_style, w_style = dense(y, cin, gain=gain, use_wscale=use_wscale, lrmul=lrmul)
            s = s + y_style
            vh.Ws = w_style
        ww = ww * tf.cast(s[:, tf.newaxis, tf.newaxis, :, tf.newaxis], w.dtype) # scale input feature maps

        # Demodulate
        if demodulate:
            d = tf.rsqrt(tf.reduce_sum(tf.square(ww), axis=[1,2,3], keepdims=True) + eps) # scaling factor
            ww = ww * d
        
        # Reshape/scale input
        if fused_modconv:
            x = tf.reshape(tf.transpose(x, [0, 3, 1, 2]), [1, -1, shape[1], shape[2]]) # [1, n*cin, h, w]
            w = tf.reshape(tf.transpose(ww, [1, 2, 3, 0, 4]), [kernel, kernel, cin, -1]) # [k, k, cin, n*cout]
            x = tf.nn.conv2d(x, tf.cast(w, x.dtype), data_format='NCHW', strides=[1,1,1,1], padding=padding)
            out_shape = x.get_shape().as_list()
            x = tf.transpose(tf.reshape(x, [-1, fmaps, out_shape[2], out_shape[3]]), [0, 2, 3, 1])
        else:
            x = x * tf.cast(s[:, tf.newaxis, tf.newaxis, :], x.dtype)
            x = tf.nn.conv2d(x, tf.cast(w, x.dtype), data_format='NHWC', strides=[1,1,1,1], padding=padding)
            if demodulate:
                x = x * tf.cast(tf.reshape(d, [-1, 1, 1, fmaps]), x.dtype)
        ret = tf.identity(x)
        ret.variables = vh
        return ret

