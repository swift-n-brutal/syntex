import tensorflow as tf
import numpy as np
import glob, os
from collections import OrderedDict

from tensorpack import (InstanceNorm, LinearWrap, Conv2D, Conv2DTranspose,
    argscope, imgaug, logger, PrintData, QueueInput, ModelSaver, Callback,
    ScheduledHyperParamSetter, PeriodicTrigger, SaverRestore, JoinData,
    AugmentImageComponent, ImageFromFile, BatchData, MultiProcessRunner,
    MergeAllSummaries, BatchNorm)
from tensorpack.tfutils.summary import add_moving_summary, add_tensor_summary

from aparse import ArgParser
from syntex.texture_utils import build_texture_loss, build_gram
from model import SynTexModelDesc, SynTexTrainer, RandomZData

MAX_EPOCH = 200
STEPS_PER_EPOCH = 4000
#
IMAGESIZE = 224
LR = 5e-5
BETA1 = 0.5
BETA2 = 0.999
BATCH = 1
TEST_BATCH = 1

def INReLU(x, name=None):
    x = InstanceNorm('inorm', x)
    return tf.nn.relu(x, name=name)

def INLReLU(x, name=None):
    x = InstanceNorm('inorm', x)
    return tf.nn.leaky_relu(x, alpha=0.2, name=name)

def BNReLU(x, name=None):
    x = BatchNorm('bnorm', x)
    return tf.nn.relu(x, name=name)

def BNLReLU(x, name=None):
    x = BatchNorm('bnorm', x)
    return tf.nn.leaky_relu(x, alpha=0.2, name=name)

def NONReLU(x, name=None):
    return tf.nn.relu(x, name=name)

def NONLReLU(x, name=None):
    return tf.nn.leaky_relu(x, alpha=0.2, name=name)

def PadConv2D(x, chan, ksize, pad_type, activation, use_bias, name=None):
    psize = ksize - 1
    p_start = psize // 2
    p_end = psize - p_start
    if psize > 0:
        x = tf.pad(x, [[0,0], [p_start, p_end], [p_start, p_end], [0,0]], mode=pad_type)
    return Conv2D(name, x, chan, ksize, padding="VALID", activation=activation, use_bias=use_bias)

class ProgressiveSynTex(SynTexModelDesc):
    def __init__(self, args):
        super(ProgressiveSynTex, self).__init__(args)
        self._image_size = args.get("image_size") or IMAGESIZE
        # The loss is averaged among gpus. So we need to scale the lr accordingly.
        n_gpu = args.get("n_gpu") or 1
        batch_size = (args.get("batch_size") or BATCH)
        equi_batch_size = max(n_gpu, 1) * batch_size
        lr = args.get("lr") or LR
        self._lr = lr * equi_batch_size
        self._beta1 = args.get("beta1") or BETA1
        self._beta2 = args.get("beta2") or BETA2
        self._n_stage = args.get("n_stage") or 5
        assert self._n_stage == 5, "Num of stages is 5 for progressive model"
        self._n_block = args.get("n_block") or 2
        act = args.get("act", "sigmoid")
        if act == "sigmoid":
            self._act = tf.nn.sigmoid
        elif act == "tanh":
            self._act = tf.nn.tanh
        elif act == "identity":
            self._act = tf.identity
        else:
            raise ValueError("Invalid activation: " + str(type(act)))
        self._loss_scale = args.get("loss_scale", 1.)
        self._pre_act = args.get("pre_act") or False
        self._nn_upsample = not (args.get("deconv_upsample") or False)
        pad_type = args.get("pad_type") or "reflect"
        if pad_type == "zero":
            self._pad_type = "CONSTANT"
        else:
            self._pad_type = pad_type.upper()
        self._norm_type = args.get("norm_type") or "instance"
        self._act_type = args.get("act_type") or "relu"
        self._grad_ksize = args.get("grad_ksize") or 3

    def inputs(self):
        return [tf.TensorSpec((None, self._image_size, self._image_size, 3), tf.float32, 'pre_image_input'),
            tf.TensorSpec((None, self._image_size, self._image_size, 3), tf.float32, 'image_target')]

    @staticmethod
    def get_parser(ps=None):
        ps = SynTexModelDesc.get_parser(ps)
        ps = ArgParser(ps, name="progressive")
        ps.add("--image-size", type=int, default=IMAGESIZE)
        ps.add("--lr", type=float, default=LR)
        ps.add("--beta1", type=float, default=BETA1)
        ps.add("--beta2", type=float, default=BETA2)
        ps.add("--n-stage", type=int, default=5, help="This argument is fixed to 5.")
        ps.add("--n-block", type=int, default=2, help="number of res blocks in each scale.")
        ps.add("--act", type=str, default="sigmoid", choices=["sigmoid", "identity"])
        ps.add("--loss-scale", type=float, default=1.)
        ps.add("--pad-type", type=str, default="reflect", choices=["reflect", "zero", "symmetric"])
        ps.add("--grad-ksize", type=int, default=3)
        ps.add("--norm-type", type=str, default="instance", choices=["instance", "batch", "none"])
        ps.add("--act-type", type=str, default="relu", choices=["relu", "lrelu"])
        ps.add_flag("--pre-act")
        ps.add_flag("--deconv-upsample")
        ps.add("--n-gpu", type=int, default=1)
        return ps

    @staticmethod
    def build_res_block(x, name, chan, pad_type, norm_type, act_type, first=False):
        """The value of input should be after normalization but before non-linearity.
        So the non-linearity function is needed to be applied in the residual branch.

        This implementation assumes that the input and output dimensions are
        consistent. The upsampling and downsampling steps are implemented in other
        modules such as Conv2D_Transpose and Conv2d, or tile and avg_pool.
        """
        with tf.variable_scope(name):
            assert x.get_shape().as_list()[-1] == chan
            shortcut = x
            if act_type == "relu": 
                res_input = tf.nn.relu(x, "act_input")
            else:
                res_input = tf.nn.leaky_relu(x, alpha=0.2, name="act_input")
            conv_branch = (LinearWrap(res_input)
                .tf.pad([[0, 0], [1, 1], [1, 1], [0, 0]], mode=pad_type)
                .Conv2D('conv0', chan, 3, padding="VALID")
                .tf.pad([[0, 0], [1, 1], [1, 1], [0, 0]], mode=pad_type)
                .Conv2D('conv1', chan, 3, padding="VALID", activation=tf.identity)
            )()
            if norm_type == "instance":
                conv_branch = InstanceNorm("inorm", conv_branch)
            elif norm_type == "batch":
                conv_branch = BatchNorm("bnorm", conv_branch)
            else:
                conv_branch = tf.identity(conv_branch, "nonorm")
            return conv_branch + shortcut

    @ staticmethod
    def build_pre_res_block(x, name, chan, pad_type, norm_type, act_type, first=False):
        """The value of input should be after normalization but before non-linearity.
        So the non-linearity function is needed to be applied in the residual branch.

        This implementation assumes that the input and output dimensions are
        consistent. The upsampling and downsampling steps are implemented in other
        modules such as Conv2D_Transpose and Conv2d, or tile and avg_pool.
        """
        with tf.variable_scope(name):
            assert x.get_shape().as_list()[-1] == chan
            shortcut = x
            if norm_type == "instance":
                res_input = InstanceNorm("inorm", x)
            elif norm_type == "batch":
                res_input = BatchNorm("bnorm", x)
            else:
                res_input = tf.nn.identity(x, "nonorm")
            if act_type == "relu":
                res_input = tf.nn.relu(res_input, "act_input")
            else:
                res_input = tf.nn.leaky_relu(res_input, alpha=0.2, name="act_input")
            return (LinearWrap(res_input)
                .tf.pad([[0,0], [1,1], [1,1], [0,0]], mode=pad_type)
                .Conv2D("conv0", chan, 3, padding="VALID")
                .tf.pad([[0,0], [1,1], [1,1], [0,0]], mode=pad_type)
                .Conv2D("conv1", chan, 3, padding="VALID", activation=tf.identity)
            )() + shortcut

    @staticmethod
    def build_upsampling_nnconv(x, name, pad_type, scale=2, chan=None, ksize=None):
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
            return PadConv2D(x_up, chan, ksize, pad_type, tf.identity, False, "conv")

    @staticmethod
    def build_upsampling_deconv(x, name, pad_type, scale=2, chan=None, ksize=None):
        if chan is None:
            chan = x.get_shape().as_list()[-1]
        if ksize is None:
            ksize = scale*2
        return Conv2DTranspose(name, x, chan, ksize,
            strides=scale, activation=tf.identity, use_bias=False)

    def build_stage_preparation(self, image_input, gram_target, coefs, name="prep"):
        layers = list(coefs.keys())
        loss_per_layer = OrderedDict()
        with tf.name_scope(name):
            feat_input = self._vgg19.build(image_input, topmost=layers[-1])
            # NOTE loss_input should exclude the loss of the last layer, because
            # the input is not computed from the gradient of the last layer.
            # While the loss of the last layer contributes to the calculation
            # of the gradient input in the current stage.
            for k in layers:
                gram_input = build_gram(feat_input[k], name="gram_"+k)
                loss_per_layer[k] = tf.reduce_mean(tf.square(gram_input - gram_target[k]),
                        axis=[1,2], name="l2_"+k)
            if len(layers) > 1:
                loss_input = tf.add_n([coefs[k]*1./4 * loss_per_layer[k] for k in layers[:-1]],
                        name="loss_input")
            else:
                loss_input = 0.
            loss_grad = loss_input + coefs[layers[-1]]*1./4 * loss_per_layer[layers[-1]]
            grads = tf.gradients(loss_grad, [feat_input[k] for k in layers])
            grad_per_layer = OrderedDict(zip(layers, grads))
        return feat_input, loss_input, loss_per_layer, grad_per_layer

    def build_stage(self, pre_image_input, gram_target, n_loss_layer, name="stage"):
        res_block = ProgressiveSynTex.build_pre_res_block if self._pre_act \
            else ProgressiveSynTex.build_res_block
        upsample = ProgressiveSynTex.build_upsampling_nnconv if self._nn_upsample \
            else ProgressiveSynTex.build_upsampling_deconv
        if self._norm_type == "instance":
            norm = InstanceNorm
            if self._act_type == "relu":
                norm_act = INReLU
            else:
                norm_act = INLReLU
        elif self._norm_type == "batch":
            norm = BatchNorm
            if self._act_type == "relu":
                norm_act = BNReLU
            else:
                norm_act = BNLReLU
        else:
            norm = tf.identity
            if self._act_type == "relu":
                norm_act = NONReLU
            else:
                norm_act = NONLReLU
        if self._act_type == "relu":
            act = NONReLU
        else:
            act = NONLReLU
        coefs = OrderedDict()
        for k in list(SynTexModelDesc.DEFAULT_COEFS.keys())[:n_loss_layer]:
            coefs[k] = SynTexModelDesc.DEFAULT_COEFS[k]
        with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
            # extract features and gradients
            image_input = self._act(pre_image_input, name="input_"+name)
            feat_input, loss_input, loss_per_layer, grad_per_layer = \
                    self.build_stage_preparation(image_input, gram_target, coefs)
            # For an adaptive texture synthesizer, we provide gradients explicitly to
            # the synthesizer.
            #            none +
            # grad[4] conv[4] -> res[4] -> up[4] +
            #                    grad[3] conv[3] -> res[3] -> up[3] +
            #                                       grad[2] conv[2] -> res[2] -> up[2] +
            #                                                               ... ...
            #                                                 up[1] +
            #                                       grad[0] conv[0] -> res[0] -> output
            with argscope([Conv2D, Conv2DTranspose], activation=norm_act, use_bias=False):
                first = True
                for layer in reversed(feat_input):
                    if layer in grad_per_layer:
                        grad = grad_per_layer[layer]
                        chan = grad.get_shape().as_list()[-1]
                        with tf.variable_scope(layer):
                            # compute pseudo grad of current layer
                            grad = PadConv2D(grad, chan, self._grad_ksize, self._pad_type, norm_act, False, "grad_conv1")
                            grad = PadConv2D(grad, chan, self._grad_ksize, self._pad_type, tf.identity, False, "grad_conv2")
                            # merge with grad from deeper layers
                            if first:
                                delta = tf.identity(grad, name="grad_merged")
                                first = False
                            else:
                                # upsample deeper grad
                                if self._pre_act:
                                    delta = norm_act(delta, "pre_inrelu")
                                else:
                                    delta = act(delta, "pre_relu")
                                delta = upsample(delta, "up", self._pad_type, chan=chan) # not normalized nor activated
                                # add two grads
                                delta = tf.add(grad, delta, name="grad_merged")
                            if not self._pre_act:
                                delta = norm("post_inorm", delta)
                            # simulate the backpropagation procedure to next level
                            for k in range(self._n_block):
                                delta = res_block(delta, "res{}".format(k), chan,
                                        self._pad_type, self._norm_type,
                                        self._act_type, first=(k == 0))
                # output
                if self._pre_act:
                    delta = norm_act(delta, "actlast")
                else:
                    delta = act(delta, "actlast")
                delta_input = PadConv2D(delta, 3, 3, self._pad_type, tf.identity, True, "convlast")
            pre_image_output = tf.add(pre_image_input, delta_input, name="pre_image_output")
        return image_input, loss_input, loss_per_layer, pre_image_output

    def build_graph(self, pre_image_input, image_target):
        """
        Parameters
        ----------
        pre_image_input : tf.Tensor
            The value are considered as the linear value before activation.
            The activation function is defined by self._act .
        image_target : tf.Tensor
            The value are considered as the actual pixel value in [0, 255]
        """
        with tf.name_scope("preprocess"):
            image_target = image_target / 255.

        def viz(name, images):
            with tf.name_scope(name):
                im = tf.concat(images, axis=2)
                #im = tf.transpose(im, [0, 2, 3, 1])
                if self._act == tf.tanh:
                    im = (im + 1.0) * 128
                else:
                    im = im * 256
                im = tf.clip_by_value(im, 0, 255)
                im = tf.cast(im, tf.uint8, name="viz")
            tf.summary.image(name, im, max_outputs=10, collections=["image_summaries"])

        # calculate gram_target
        _, gram_target = self._build_extractor(image_target, name="ext_target")
        # inference pre_image_output from pre_image_input and gram_target
        self.image_outputs = list()
        self.losses = list()
        pre_image_output = pre_image_input
        with tf.variable_scope("syn"):
            for s in range(self._n_stage):
                image_input, loss_overall_input, _, pre_image_output = \
                    self.build_stage(pre_image_output, gram_target, s+1, name="stage%d" % s)
                self.image_outputs.append(image_input)
                self.losses.append(tf.reduce_mean(loss_overall_input, name="loss%d" % s))
        self.collect_variables("syn")
        #
        image_output = self._act(pre_image_output, name="output")
        loss_overall_output, loss_layer_output, _ = \
            self._build_loss(image_output, gram_target, calc_grad=False)
        self.image_outputs.append(image_output)
        self.losses.append(tf.reduce_mean(loss_overall_output, name="loss_output"))
        self.loss_layer_output = loss_layer_output
        self.loss_layer_output = OrderedDict()
        with tf.name_scope("loss_layer_output"):
            for layer in loss_layer_output:
                self.loss_layer_output[layer] = tf.reduce_mean(loss_layer_output[layer], name=layer)
        # average losses from all stages
        weights = [1.]
        for _ in range(len(self.losses) - 1):
            weights.append(weights[-1] * self._loss_scale)
        # skip the first loss as it is computed from noise
        self.loss = tf.add_n([weights[i] * loss \
            for i, loss in enumerate(reversed(self.losses[1:]))], name="loss")
        # summary
        viz("stages-target", self.image_outputs + [image_target])
        add_moving_summary(self.loss, *self.losses, *self.loss_layer_output.values())

    def optimizer(self):
        lr_var = tf.get_variable("learning_rate", initializer=self._lr, trainable=False)
        add_tensor_summary(lr_var, ['scalar'], main_tower_only=True)
        return tf.train.AdamOptimizer(lr_var, beta1=self._beta1, beta2=self._beta2, epsilon=1e-3)


def get_data(datadir, size=IMAGESIZE, isTrain=True, zmin=-1, zmax=1):
    if isTrain:
        augs = [
            imgaug.ResizeShortestEdge(int(size*1.143)),
            imgaug.RandomCrop(size),
            imgaug.Flip(horiz=True),
        ]
    else:
        augs = [
            imgaug.ResizeShortestEdge(int(size*1.143)),
            imgaug.CenterCrop(size)]


    def get_images(dir):
        files = sorted(glob.glob(os.path.join(dir, "*.jpg")))
        df = ImageFromFile(files, channel=3, shuffle=isTrain)
        random_df = RandomZData([size, size, 3], zmin, zmax)
        return JoinData([random_df, AugmentImageComponent(df, augs)])
    
    names = ['train']  if isTrain else ['test']
    df = get_images(*[os.path.join(datadir, n) for n in names])
    df = BatchData(df, BATCH if isTrain else TEST_BATCH)
    return df


class VisualizeTestSet(Callback):
    def __init__(self, data_folder, image_size):
        self._data_folder = data_folder
        self._image_size = image_size

    def _setup_graph(self):
        self.pred = self.trainer.get_predictor(
            ["pre_image_input", "image_target"], ['stages-target/viz'])

    def _before_train(self):
        self.val_ds = get_data(self._data_folder, self._image_size, isTrain=False)
        self.val_ds.reset_state()

    def _trigger(self):
        idx = 0
        for pii, it in self.val_ds:
            print("------------------ predict --------------")
            print(pii.shape, pii.dtype)
            print(it.shape, it.dtype)
            viz = self.pred(pii, it)
            self.trainer.monitors.put_image('test-{}'.format(idx), viz)
            idx += 1


if __name__ == "__main__":
    ps = ProgressiveSynTex.get_parser()
    ps.add("--data-folder", type=str, default="../images/scaly")
    ps.add("--save-folder", type=str, default="train_log/impr")
    ps.add("--steps-per-epoch", type=int, default=STEPS_PER_EPOCH)
    ps.add("--max-epoch", type=int, default=MAX_EPOCH)
    ps.add("--save-epoch", type=int, help="Save parameters every n epochs")
    ps.add("--image-steps", type=int, help="Synthesize images every n steps")
    ps.add("--scalar-steps", type=int, help="Period to add scalar summary", default=0)
    ps.add("--batch-size", type=int)
    args = ps.parse_args()
    print("Arguments")
    ps.print_args()
    print()

    data_folder = args.get("data_folder")
    save_folder = args.get("save_folder")
    image_size = args.get("image_size")
    max_epoch = args.get("max_epoch")
    save_epoch = args.get("save_epoch") or max_epoch // 10
    # Scale lr and steps_per_epoch accordingly.
    # Make sure the total number of gradient evaluations is consistent.
    n_gpu = args.get("n_gpu") or 1
    batch_size = (args.get("batch_size") or BATCH)
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
    zmin, zmax = (0, 1) if args.get("act") == "identity" else (-1, 1)

    if save_folder == None:
        logger.auto_set_dir()
    else:
        logger.set_logger_dir(save_folder)

    df = get_data(data_folder, image_size, zmin=zmin, zmax=zmax)
    df = PrintData(df)
    data = QueueInput(df)

    SynTexTrainer(data, ProgressiveSynTex(args), n_gpu).train_with_defaults(
        callbacks=[
            PeriodicTrigger(ModelSaver(), every_k_epochs=save_epoch),
            PeriodicTrigger(ModelSaver(), every_k_epochs=end_epoch), # save model at last
            ScheduledHyperParamSetter(
                'learning_rate',
                [(start_dec_epoch, lr), (max_epoch, 0)], interp="linear"),
            #PeriodicTrigger(VisualizeTestSet(data_folder, image_size), every_k_epochs=10),
            MergeAllSummaries(period=scalar_steps), # scalar only
            MergeAllSummaries(period=image_steps, key="image_summaries"),
        ],
        max_epoch= end_epoch,
        steps_per_epoch=steps_per_epoch,
        session_init=None
    )
