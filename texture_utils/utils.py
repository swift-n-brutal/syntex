import tensorflow as tf
import numpy as np
from collections import OrderedDict

NP_DTYPE = np.float32
TF_DTYPE = tf.float32

def build_gram(feat, mask=None, eps=1e-6, name="gram"):
    with tf.name_scope(name):
        shape = feat.shape.as_list()
        if mask is not None:
            mask = tf.expand_dims(mask, -1)
            feat = tf.multiply(feat, mask, name="masked_feat")
            normalizer = tf.reduce_sum(mask, axis=[1,2,3], keep_dims=True)
        else:
            normalizer = NP_DTYPE(np.prod(shape[1:-1]))
        feat_reshaped = tf.reshape(feat, [tf.shape(feat)[0], -1, shape[-1]])
        return tf.divide(
            tf.matmul(feat_reshaped, feat_reshaped, transpose_a=True),
            normalizer+NP_DTYPE(eps)
        )

def build_texture_loss(a, b, coefs, is_gram_a=False, is_gram_b=True,
        calc_grad=False, name='texture_loss'):
    """a is considered as the input, b is considered as the target.
    """
    with tf.name_scope(name):
        if is_gram_a:
            gram_a = a
        else:
            gram_a = OrderedDict()
            for k in a:
                with tf.name_scope(k):
                    gram_a[k] = build_gram(a[k])
        
        if is_gram_b:
            gram_b = b
        else:
            gram_b = OrderedDict()
            for k in b:
                with tf.name_scope(k):
                    gram_b[k] = build_gram(b[k])
        
        loss_layer = OrderedDict()
        for k in gram_a:
            gram_diff = gram_a[k] - gram_b[k]
            loss_layer[k] = tf.reduce_mean(tf.square(gram_diff),
                    axis=[1,2], name="l2_"+k)
        loss_overall = tf.add_n([coefs[k]*1./4 * loss_layer[k] for k in loss_layer], name="loss_overall")
    if calc_grad:
        grad_layer = OrderedDict()
        for k in gram_a:
            # NOTE tf.gradients returns a list of grads w.r.t. inputs
            grad_layer[k] = tf.gradients(loss_layer[k], a[k])[0]
        return loss_overall, loss_layer, grad_layer
    else:
        return loss_overall, loss_layer, None
