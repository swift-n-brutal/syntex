import tensorflow as tf
import numpy as np
from collections import OrderedDict
import glob
import os

from tensorpack import (InstanceNorm, BatchNorm, Conv2DTranspose,
    LinearWrap, layer_register, imgaug, PrintData, QueueInput, ModelSaver,
    Callback, ScheduledHyperParamSetter, PeriodicTrigger, SaverRestore,
    JoinData, AugmentImageComponent, ImageFromFile, BatchData, MultiProcessRunner,
    MergeAllSummaries, PredictConfig, OfflinePredictor, SmartInit,
    argscope, logger, regularize_cost, VariableHolder)
from tensorpack.tfutils.summary import (add_moving_summary,
        add_tensor_summary, add_activation_summary, add_param_summary)
from tensorpack.tfutils.tower import get_current_tower_context

from syntex.aparse import ArgParser
from syntex.texture_utils import build_texture_loss, build_gram
from model import SynTexModelDesc, SynTexTrainer, RandomZData

from layers import mod_conv2d, get_bias

MAX_EPOCH = 200
STEPS_PER_EPOCH = 4000
#
IMAGE_SIZE = 256
LR = 2e-4
BETA1 = 0.5
BETA2 = 0.999
EPSILON = 1e-3
N_BLOCK = 2
BATCH = 1
TEST_BATCH = 1

def act(name, x, norm_type, alpha, gain=None):
    with tf.variable_scope(name, default_name="act"):
        if alpha is None:
            x = tf.identity(x)
        elif alpha == 0:
            x = tf.nn.relu(x)
        else:
            x = tf.nn.leaky_relu(x, alpha=alpha)
        if gain is None:
            gain = 1. if alpha is None else np.sqrt(2)
        if gain != 1:
            x = x * tf.constant(gain, dtype=x.dtype)
    return x

class ActFactory():
    def __init__(self, norm_type, alpha):
        self._norm_type = norm_type
        self._alpha = alpha
    
    def __call__(self, x, name=None):
        return act(name, x, self._norm_type, self._alpha)

@layer_register(log_shape=True)
def Conv2D(inputs, filters, kernel_size,
        demodulate=True,
        gain=1,
        use_wscale=True,
        lrmul=1,
        fused_modconv=True,
        eps=1e-8,
        padding="SAME",
        activation=None,
        use_bias=True):
    x = mod_conv2d(inputs, None, filters, kernel_size,
            demodulate=demodulate, gain=gain, use_wscale=use_wscale, lrmul=lrmul,
            fused_modconv=fused_modconv, eps=eps, padding=padding) 
    vh = x.variables
    if use_bias:
        b = get_bias(filters, name='b')
        vh.b = b
        x = tf.nn.bias_add(x, b)
    if activation is not None:
        x = activation(x)
    ret = tf.identity(x, name='output')
    ret.variables = vh
    return ret

@layer_register(log_shape=True)
def SphericalAdd(x1, x2, theta_mean=0., theta_std=0.,
        use_wscale=True, lrmul=1., adaptive_lr=True, channelwise=True):
    """y = x1 * cos(theta) + x2 * sin(theta)
    Special cases:
        y = x1 if theta == 0
        y = x2 if theta == np.pi/2
    """
    chan = x1.get_shape().as_list()[-1] if channelwise else 1
    theta = get_bias(chan, base_std=theta_std, use_wscale=use_wscale,
            lrmul=lrmul, adaptive_lr=adaptive_lr, name="theta")
    vh = VariableHolder(theta=theta)
    theta = theta + theta_mean
    s1 = tf.math.cos(theta, name="s1")
    s2 = tf.math.sin(theta, name="s2")
    ret = tf.identity(tf.add(x1*s1, x2*s2), name="output")
    ret.variables = vh
    return ret

def pad_conv2d(name, x, chan, ksize, pad_type, **kwargs):
    # Set activation and use_bias explicitly if needed
    with tf.variable_scope(name):
        psize = ksize - 1
        p_start = psize // 2
        p_end = psize - p_start
        if psize > 0:
            x = tf.pad(x, [[0,0], [p_start, p_end], [p_start, p_end], [0,0]], mode=pad_type, name="pad")
        return Conv2D("conv2d", x, chan, ksize, padding="VALID", **kwargs)

def res_block(name, x, chan, pad_type, norm_type, alpha=0.2, bottleneck=False, pre_act=False, gain=1./np.sqrt(2.)):
    """
    The value of input should be
        * after normalization but before non-linearity (if pre_act is False).
        * before normalization and non-linearity (if pre_act is True).
    So the activation function is needed to be applied in the residual branch.

    This implementation assumes that the input and output dimensions are
    consistent. The upsampling and downsampling steps are implemented in other
    modules such as Conv2D_Transpose and Conv2d, or tile and avg_pool.
    """
    with tf.variable_scope(name):
        assert x.get_shape().as_list()[-1] == chan
        shortcut = x
        conv_branch = act("act_input", x, "none", alpha)
        if bottleneck:
            assert chan % 4 == 0, chan
            chan_b = chan // 4
            conv_branch = pad_conv2d("conv0", conv_branch, chan_b, 1, pad_type) # default activation in argscope
            conv_branch = pad_conv2d("conv1", conv_branch, chan_b, 3, pad_type)
            conv_branch = pad_conv2d("conv2", conv_branch, chan, 1, pad_type, activation=tf.identity)
        else:
            conv_branch = pad_conv2d("conv0", conv_branch, chan, 3, pad_type) # default activation in argscope
            conv_branch = pad_conv2d("conv1", conv_branch, chan, 3, pad_type, activation=tf.identity)
        output = shortcut + conv_branch
        if gain is None:
            return output
        else:
            return output * tf.constant(gain, dtype=output.dtype)
        
def upsampling_nnconv(name, x, pad_type, scale=2, chan=None, ksize=None):
    _, h, w, c = x.get_shape().as_list()
    if chan is None:
        chan = c
    if ksize is None:
        ksize = scale*2 - 1
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        if scale == 1:
            x_up = x
        else:
            assert scale > 1
            x = tf.reshape(x, [-1, h, 1, w, 1, c])
            x_up = tf.tile(x, [1, 1, scale, 1, scale, 1], name="nn")
            x_up = tf.reshape(x_up, [-1, h*scale, w*scale, c])
        return pad_conv2d("conv", x_up, chan, ksize, pad_type,
            activation=tf.identity, use_bias=False)

def upsampling_deconv(name, x, pad_type, scale=2, chan=None, ksize=None):
    if chan is None:
        chan = x.get_shape().as_list()[-1]
    if ksize is None:
        ksize = scale*2
    raise NotImplementedError
    return Conv2DTranspose(name, x, chan, ksize,
        strides=scale, activation=tf.identity, use_bias=False)

def get_relu_gate(name, x, alpha=0.):
    with tf.name_scope(name):
        return tf.where_v2(tf.greater(x, 0.),
            tf.constant(1., dtype=tf.float32), tf.constant(alpha, dtype=tf.float32))

class Style2PO(SynTexModelDesc):
    GATE_SOURCE = OrderedDict([
        ("conv1_1", "conv1_1"),
        ("pool1", "conv1_2"),
        ("pool2", "conv2_2"),
        ("pool3", "conv3_4"),
        ("pool4", "conv4_4")
    ])
    N_BLOCK_BASE = OrderedDict([
        ("conv1_1", 1),
        ("pool1", 1),
        ("pool2", 2),
        ("pool3", 4),
        ("pool4", 4)
    ])
    def __init__(self, args):
        super(Style2PO, self).__init__(args)
        self._image_size = args.get("image_size") or IMAGE_SIZE
        n_gpu = args.get("n_gpu") or 1
        batch_size = args.get("batch_size") or BATCH
        equi_batch_size = max(n_gpu, 1) * batch_size
        self._lr = args.get("lr") or LR
        self._beta1 = args.get("beta1") or BETA1
        self._beta2 = args.get("beta2") or BETA2
        self._epsilon = args.get("epsilon") or EPSILON
        self._loss_scale = args.get("loss_scale" or 1.)
        self._top_stage = args.get("top_stage") or 5
        self._n_stage = args.get("n_stage") or 1
        self._n_block = args.get("n_block") or N_BLOCK
        pad_type = args.get("pad_type") or "zero"
        if pad_type == "zero":
            self._pad_type = "CONSTANT"
        else:
            self._pad_type = pad_type.upper()
        self._norm_type = args.get("norm_type") or "instance"
        act_type = args.get("act_type") or "lrelu"
        if act_type == "lrelu":
            self._alpha = 0.2
        elif act_type == "relu":
            self._alpha = 0
        else:
            self._alpha = None
        act_input = args.get("act_input") or "sigmoid"
        if act_input == "sigmoid":
            self._act_input = tf.nn.sigmoid
        elif act_input == "identity":
            self._act_input = tf.identity
        else:
            raise ValueError("Invalid activation to input: " + str(act_input))
        self._grad_ksize = args.get("grad_ksize") or 3
        self._theta_mean = args.get("theta_mean") or np.pi*0.25
        self._theta_lrmul = args.get("theta_lrmul") or 0.1
        self._pre_act = args.get("pre_act") or False
        self._bottleneck = args.get("bottleneck") or False
        self._deconv = args.get("deconv") or False
        self._gate = args.get("gate") or False
        self._same_block = args.get("same_block") or False
        self._stop_grad = args.get("stop_grad") or False
        
    @staticmethod
    def get_parser(ap=None):
        ap = super(Style2PO, Style2PO).get_parser(ap)
        ap = ArgParser(ap, name="style-po")
        ap.add("--image-size", type=int, default=IMAGE_SIZE)
        ap.add("--lr", type=float, default=LR)
        ap.add("--beta1", type=float, default=BETA1)
        ap.add("--beta2", type=float, default=BETA2)
        ap.add("--epsilon", type=float, default=EPSILON)
        ap.add("--loss-scale", type=float, default=1.,
            help="Multiplier of the weights for losses from different stages. The weight for the last stage is 1 and the other stage is {scale}^{stage}.")
        ap.add("--top-stage", type=int, default=5, help="The number of stages.")
        ap.add("--n-stage", type=int, default=1, help="The number of repeats in each stage.")
        ap.add("--n-block", type=int, default=N_BLOCK, help="number of res blocks in each scale")
        ap.add("--norm-type", type=str, default="batch", choices=["instance", "batch", "none"],
            help="BatchNorm has a slight performance gain (~10%%) over InstanceNorm when batch_size is 1.")
        ap.add("--pad-type", type=str, default="zero", choices=["zero", "reflect", "symmetric"])
        ap.add("--act-type", type=str, default="lrelu", choices=["lrelu", "relu", "none"])
        ap.add("--act-input", type=str, default="sigmoid", choices=["sigmoid", "identity"],
            help="Apply act_input to the input image before feeding it to the network")
        ap.add("--grad-ksize", type=int, default=3,
            help="Size of kernel used to convolve with per-layer gradient")
        ap.add("--theta-mean", type=float, default=np.pi*0.25,
                help="Initial value of the angle in SphericalAdd")
        ap.add("--theta-lrmul", type=float, default=0.1)
        ap.add_flag("--pre-act", help="Pre-act residual block")
        ap.add_flag("--bottleneck", help="Bottleneck residual block")
        ap.add_flag("--deconv", help="Use deconv upsampling. Default is nearest neighbor upsampling")
        ap.add_flag("--gate", help="Mask the backward signal with the activation state")
        ap.add_flag("--same-block", help="Use the same n_block in each stage. Otherwise use n_block * n_block_base[layer].")
        ap.add_flag("--stop-grad")
        return ap

    def inputs(self):
        return [tf.TensorSpec((None, self._image_size, self._image_size, 3), tf.float32, "image_input"),
            tf.TensorSpec((None, self._image_size, self._image_size, 3), tf.float32, "image_target")]

    def build_stage_preparation(self, x, gram_target, coefs, grad_gain=10., name="prep"):
        """
        Parameters
        ----------
        x : The input image
        gram_target : The dict of target gram matrices
        coefs : The dict of coefs whose keys are the layers that contribute to the texture loss
        grad_gain : The gain multiplier of the per-layer gradient. To deal with images
            of arbitrary sizes, we manually compute the per-layer gradients by removing
            normalization w.r.t. the size. A side effect is that the magnitude of input
            gradient to the PO network may not be suitable for training. This value is
            emprically tuned for DTD dataset.

        # NOTE loss_input should exclude the loss of the last layer, because
        # the input is not computed from the gradient of the last layer,
        # while the loss of the last layer contributes to the calculation
        # of the per-layer gradient in the current stage.
        """
        layers = list(coefs.keys())
        loss_per_layer = OrderedDict()
        grad_per_layer = OrderedDict()
        with tf.name_scope(name):
            feat = self._vgg19.build(x, topmost=layers[-1])
            for k in layers:
                n, h, w, chan = feat[k].get_shape().as_list()
                gram = build_gram(feat[k], name="gram_"+k)
                diff_gram = gram - gram_target[k]
                if k != layers[-1]:
                    loss_per_layer[k] = tf.reduce_mean(tf.square(diff_gram),
                            axis=[1, 2], name="l2_"+k) * (1./4)
                grad = tf.reshape(
                        tf.matmul(tf.reshape(feat[k], [-1, h*w, chan]), diff_gram),
                        (-1, h, w, chan), name="grad_"+k)
                if self._stop_grad:
                    grad = tf.stop_gradient(grad)
                grad_per_layer[k] = grad * tf.constant(grad_gain/(chan*chan), dtype=feat[k].dtype)
            if len(layers) > 1:
                loss_input = tf.add_n([coefs[k] * loss_per_layer[k] for k in layers[:-1]], name="loss_input")
            else:
                loss_input = 0.
            return feat, loss_input, loss_per_layer, grad_per_layer
        """ 2020-08-07 Counteract the influence of image size
        layers = list(coefs.keys())
        loss_per_layer = OrderedDict()
        with tf.name_scope(name):
            feat = self._vgg19.build(x, topmost=layers[-1])
            for k in layers:
                gram = build_gram(feat[k], name="gram_"+k)
                diff_gram = gram - gram_target[k]
                loss_per_layer[k] = tf.reduce_mean(tf.square(diff_gram),
                    axis=[1,2], name="l2_"+k) * (1./4)
            if len(layers) > 1:
                loss_input = tf.add_n([coefs[k] * loss_per_layer[k] for k in layers[:-1]],
                    name="loss_input")
            else:
                loss_input = 0.
            loss_grad = loss_input + coefs[layers[-1]] * loss_per_layer[layers[-1]]
            #loss_grad = tf.add_n([loss_per_layer[k] for k in layers], name="loss_grad")
            grads = tf.gradients(loss_grad, [feat[k] for k in layers])
            if self._stop_grad:
                grads = [tf.stop_gradient(_g) for _g in grads]
            #grad_per_layer = OrderedDict(zip(layers, grads))
            # TODO Counteract the influence of image size
            grad_per_layer = OrderedDict()
            for i, k in enumerate(layers):
                size = np.prod(feat[k].get_shape().as_list()[1:-1])
                grad_per_layer[k] = grads[i] * tf.constant(size, dtype=grads[i].dtype)
        return feat, loss_input, loss_per_layer, grad_per_layer
        """

    def build_stage(self, x, gram_target, coefs, gain=1/np.sqrt(2), name="stage"):
        acti = ActFactory(self._norm_type, self._alpha)
        upsample = upsampling_deconv if self._deconv else upsampling_nnconv
        first = True
        with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
            # extract features and gradients
            x_image = self._act_input(x, name="input_"+name)
            feat, loss_input, loss_per_layer, grad_per_layer = \
                self.build_stage_preparation(x_image, gram_target, coefs)
            #            none +
            # grad[4] conv[4] -> res[4] -> up[4] +
            #                    grad[3] conv[3] -> res[3] -> up[3] +
            #                                       grad[2] conv[2] -> res[2] -> up[2] +
            #                                                               ... ...
            #                                                 up[1] +
            #                                       grad[0] conv[0] -> res[0] -> output
            with argscope([Conv2D, Conv2DTranspose], activation=acti, use_bias=False):
                for layer in reversed(feat):
                    if layer in grad_per_layer:
                        with tf.variable_scope(layer):
                            # compute pseudo grad of current layer
                            grad = tf.identity(grad_per_layer[layer], name="input")
                            add_activation_summary(grad, types=["rms", "histogram"], collections=["acti_summaries"])
                            chan = grad.get_shape().as_list()[-1]
                            grad = pad_conv2d("grad_conv", grad, chan, self._grad_ksize,
                                self._pad_type, activation=tf.identity)
                            add_activation_summary(grad, types=["rms", "histogram"], collections=["acti_summaries"])
                            # merge with grad from deeper layers
                            if first:
                                delta = tf.identity(grad, name="grad_merged")
                                first = False
                            else:
                                # change chan of delta
                                delta = pad_conv2d("conv_chan", delta, chan, 3,
                                    self._pad_type, activation=tf.identity)
                                #delta = tf.add(grad, delta, name="grad_merged") * gain
                                delta = SphericalAdd("grad_merged", delta, grad, self._theta_mean, lrmul=self._theta_lrmul)
                            # upsample
                            if layer != "conv1_1":
                                delta = upsample("up", delta, self._pad_type, chan=chan) # no activated
                            #-------------------
                            # add relu gate here
                            if self._gate:
                                gate = get_relu_gate("gate", feat[Style2PO.GATE_SOURCE[layer]], 0.)
                                assert gate.get_shape().as_list() == delta.get_shape().as_list(),\
                                    "{} vs {}".format(gate.get_shape().as_list(), delta.get_shape().as_list())
                                delta = delta * gate
                            #-------------------
                            # simulate the backpropagation to next level
                            if self._same_block:
                                n_block = self._n_block
                            else:
                                n_block = self._n_block * Style2PO.N_BLOCK_BASE[layer]
                            for k in range(n_block):
                                delta = res_block("res{}".format(k), delta, chan,
                                    self._pad_type, self._norm_type, self._alpha,
                                    self._bottleneck, self._pre_act)
                            delta = acti(delta, "acti_output")
                # output
                delta_x = pad_conv2d("conv_last", delta, 3, 1, self._pad_type,
                        demodulate=False, activation=tf.identity, use_bias=True)
            if self._stop_grad:
                x = tf.add(tf.stop_gradient(x), delta_x, name="output")
            else:
                x = tf.add(x, delta_x, name="output")
        return x_image, loss_input, loss_per_layer, x

    def build_graph(self, x, image_target):
        with tf.name_scope("preprocess"):
            image_target = image_target / 255.
        
        def viz(name, images):
            with tf.name_scope(name):
                im = tf.concat(images, axis=2)
                #im = tf.transpose(im, [0, 2, 3, 1])
                if self._act_input == tf.tanh:
                    im = (im + 1.0) * 127.5
                else:
                    im = im * 255
                im = tf.clip_by_value(im, 0, 255)
                im = tf.round(im)
                im = tf.cast(im, tf.uint8, name="viz")
            return im

        # calculate gram_target
        _, gram_target = self._build_extractor(image_target, name="ext_target")
        # inference pre_image_output from pre_image_input and gram_target
        self.image_outputs = list()
        self.loss_per_stage = list()
        x_output = x
        with tf.variable_scope("syn"):
            # use data stats in both train and test phases
            with argscope(BatchNorm, training=True):
                for s in range(self._top_stage):
                    # get the first (s+1) coefs
                    coefs = OrderedDict()
                    for k in list(SynTexModelDesc.DEFAULT_COEFS.keys())[:s+1]:
                        coefs[k] = SynTexModelDesc.DEFAULT_COEFS[k]
                    for repeat in range(self._n_stage):
                        x_image, loss_input, _, x_output = \
                            self.build_stage(x_output, gram_target, coefs, name="stage%d-%d" % (s, repeat))
                        if repeat == 0:
                            self.image_outputs.append(x_image)
                            self.loss_per_stage.append(tf.reduce_mean(loss_input, name="loss%d" % s))
        self.collect_variables("syn")
        #
        image_output = self._act_input(x_output, name="output")
        loss_output, loss_per_layer_output, _ = \
            self._build_loss(image_output, gram_target, calc_grad=False)
        self.image_outputs.append(image_output)
        self.loss_per_stage.append(tf.reduce_mean(loss_output, name="loss_output"))
        self.loss_per_layer_output = OrderedDict()
        with tf.name_scope("loss_per_layer_output"):
            for layer in loss_per_layer_output:
                self.loss_per_layer_output[layer] = tf.reduce_mean(loss_per_layer_output[layer], name=layer)
        # average losses from all stages
        weights = [1.]
        for _ in range(len(self.loss_per_stage) - 1):
            weights.append(weights[-1] * self._loss_scale)
        # skip the first loss as it is computed from noise
        self.loss = tf.add_n([weights[i] * loss \
            for i, loss in enumerate(reversed(self.loss_per_stage[1:]))], name="loss")
        wd_cost = regularize_cost(".*/W", tf.nn.l2_loss, name="wd_cost")
        # summary
        #with tf.device("/cpu:0"):
        stages_target = viz("stages-target", self.image_outputs + [image_target])
        ctx = get_current_tower_context()
        if ctx is not None and ctx.is_main_training_tower:
            tf.summary.image("stages-target", stages_target, max_outputs=10, collections=["image_summaries"])
            add_moving_summary(self.loss, wd_cost, *self.loss_per_stage, *self.loss_per_layer_output.values())
            add_param_summary(('.*/theta', ['histogram']), collections=["acti_summaries"])

    def optimizer(self):
        lr_var = tf.get_variable("learning_rate", initializer=self._lr, trainable=False)
        add_tensor_summary(lr_var, ['scalar'], main_tower_only=True)
        return tf.train.AdamOptimizer(lr_var, beta1=self._beta1, beta2=self._beta2, epsilon=self._epsilon)

def get_data(datadir, size=IMAGE_SIZE, isTrain=True, zmin=-1, zmax=1, batch=BATCH, shuffle_read=False):
    if isTrain:
        augs = [
            imgaug.ResizeShortestEdge(int(size*1.143)),
            imgaug.RandomCrop(size),
            imgaug.Flip(horiz=True),
        ]
    else:
        augs = [
            imgaug.ResizeShortestEdge(int(size*1.143)),
            imgaug.CenterCrop(size),
        ]

    def get_images(dir):
        files = glob.glob(os.path.join(dir, "*.jpg"))
        if shuffle_read:
            import random
            random.seed(1)
            random.shuffle(files)
        else:
            files = sorted(files)
        image_df = ImageFromFile(files, channel=3, shuffle=isTrain)
        image_df = AugmentImageComponent(image_df, augs)
        random_df = RandomZData([size, size, 3], zmin, zmax)
        return JoinData([random_df, image_df])
    
    names = ['train']  if isTrain else ['test']
    df = get_images(*[os.path.join(datadir, n) for n in names])
    df = BatchData(df, batch)
    return df

class VisualizeTestSet(Callback):
    def __init__(self, datadir, size=IMAGE_SIZE, zmin=-1, zmax=1, batch=TEST_BATCH, max_num=100):
        self._datadir = datadir
        self._size = size
        self._zmin = zmin
        self._zmax = zmax
        self._batch = batch
        self._max_num = max_num

    def _setup_graph(self):
        self.pred = self.trainer.get_predictor(
            ["image_input", "image_target"], ['stages-target/viz'])

    def _before_train(self):
        self.val_ds = get_data(self._datadir, self._size, False,
            self._zmin, self._zmax, self._batch, True)
        self.val_ds.reset_state()

    def _trigger(self):
        idx = 0
        for rz, it in self.val_ds:
            if idx >= self._max_num:
                break
            viz, = self.pred(rz, it)
            self.trainer.monitors.put_image('test-{}'.format(idx), viz)
            idx += 1

def get_test_batch(batch_size, train_ds, test_data):
    # Insert the test data into a batch of train data.
    train_ds.reset_state()
    train_data = next(iter(train_ds))
    assert train_data[0].shape[0] == batch_size, "{}".format(train_data[0].shape[0])
    for i, data in enumerate(train_data):
        data[i][0, ...] = test_data[i][0, ...]
    return train_data

def test(args):
    from imageio import imsave
    from tictoc import Timer
    data_folder = args.get("data_folder")
    image_size = args.get("image_size")
    batch_size = args.get("batch_size") or BATCH
    test_ckpt = args.get("test_ckpt")
    test_folder = args.get("test_folder")
    if not os.path.exists(test_folder):
        os.makedirs(test_folder)
    pred_config = PredictConfig(
        model=Style2PO(args),
        session_init=SmartInit(test_ckpt),
        input_names=["image_input", "image_target"],
        output_names=['stages-target/viz', 'loss_output']
    )
    predictor = OfflinePredictor(pred_config)
    zmin, zmax = (0, 1) if args.get("act_input") == "identity" else (-1, 1)
    test_ds = get_data(data_folder, image_size, False,
            zmin, zmax, batch_size)
    test_ds.reset_state()
    idx = 0
    losses = list()
    print("------------------ predict --------------")
    timer = Timer("predict", tic=True, show=Timer.STDOUT)
    for rz, it in test_ds:
        output_array, loss_output = predictor(rz, it)
        if output_array.ndim == 4:
            for i in range(output_array.shape[0]):
                imsave(os.path.join(test_folder, "test-{}.jpg".format(idx)), output_array[i])
                idx += 1
        else:
            imsave(os.path.join(test_folder, "test-{}.jpg".format(idx)), output_array)
            idx += 1
        losses.append(loss_output)
        print("loss #", idx, "=", loss_output)
    timer.toc(Timer.STDOUT)
    print("Test and save", idx, "images to", test_folder, "avg loss =", np.mean(losses))

def train(args):    
    data_folder = args.get("data_folder")
    save_folder = args.get("save_folder")
    image_size = args.get("image_size")
    max_epoch = args.get("max_epoch")
    save_epoch = args.get("save_epoch") or max_epoch // 10
    # Scale lr and steps_per_epoch accordingly.
    # Make sure the total number of gradient evaluations is consistent.
    n_gpu = args.get("n_gpu") or 1
    batch_size = args.get("batch_size") or BATCH
    equi_batch_size = max(n_gpu, 1) * batch_size
    lr = args.get("lr") or LR
    lr *= equi_batch_size
    steps_per_epoch = args.get("steps_per_epoch") or 1000
    steps_per_epoch /= equi_batch_size
    image_steps = args.get("image_steps") or steps_per_epoch // 10
    scalar_steps = args.get("scalar_steps")
    if scalar_steps > 0:
        scalar_steps = max(scalar_steps // equi_batch_size, 1)
    else:
        scalar_steps = 0 # merge scalar summary every epoch
    # lr starts decreasing at half of max epoch
    start_dec_epoch = max_epoch // 2
    # stops when lr is 0.01 of its initial value
    end_epoch = max_epoch - int((max_epoch - start_dec_epoch) * 0.01)
    # adjust noise input range according to the input act
    zmin, zmax = (0, 1) if args.get("act_input") == "identity" else (-1, 1)

    if save_folder == None:
        logger.auto_set_dir()
    else:
        logger.set_logger_dir(save_folder)

    df = get_data(data_folder, image_size, zmin=zmin, zmax=zmax, batch=batch_size)
    df = PrintData(df)
    data = QueueInput(df)

    SynTexTrainer(data, Style2PO(args), n_gpu).train_with_defaults(
        callbacks=[
            PeriodicTrigger(ModelSaver(), every_k_epochs=save_epoch),
            PeriodicTrigger(ModelSaver(), every_k_epochs=end_epoch), # save model at last
            ScheduledHyperParamSetter(
                'learning_rate',
                [(start_dec_epoch, lr), (max_epoch, 0)], interp="linear"),
            PeriodicTrigger(VisualizeTestSet(data_folder, image_size), every_k_epochs=max(1, max_epoch // 100)),
            #MergeAllSummaries(period=scalar_steps), # scalar only, slowdown in training, use TCMalloc
            MergeAllSummaries(period=image_steps, key="image_summaries"),
            MergeAllSummaries(key="acti_summaries"),
        ],
        max_epoch=end_epoch,
        steps_per_epoch=steps_per_epoch,
        session_init=None
    )

if __name__ == "__main__":
    ap = Style2PO.get_parser()
    ap.add("--data-folder", type=str, default="../images/scaly")
    ap.add("--save-folder", type=str, default="train_log/mod")
    ap.add("--steps-per-epoch", type=int, default=STEPS_PER_EPOCH)
    ap.add("--max-epoch", type=int, default=MAX_EPOCH)
    ap.add("--save-epoch", type=int, help="Save parameters every n epochs")
    ap.add("--image-steps", type=int, help="Synthesize images every n steps")
    ap.add("--scalar-steps", type=int, help="Period to add scalar summary", default=0)
    ap.add("--batch-size", type=int)
    ap.add_flag("--test-only")
    ap.add("--test-folder", type=str, default="test_log")
    ap.add("--test-ckpt", type=str)
    args = ap.parse_args()
    print("Arguments")
    ap.print_args()
    print()

    if args.get("test_only"):
        test(args)
    else:
        train(args)
