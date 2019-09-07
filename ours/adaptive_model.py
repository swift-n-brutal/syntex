import tensorflow as tf
import numpy as np
import glob, os
from collections import OrderedDict

from tensorpack import (InstanceNorm, LinearWrap, Conv2D, Conv2DTranspose,
    argscope, imgaug, logger, PrintData, QueueInput, ModelSaver, Callback,
    ScheduledHyperParamSetter, PeriodicTrigger, SaverRestore, JoinData,
    AugmentImageComponent, ImageFromFile, BatchData, MultiProcessRunner,
    MergeAllSummaries)
from tensorpack.tfutils.summary import add_moving_summary

from aparse import ArgParser
from syntex.texture_utils import build_texture_loss
from model import SynTexModelDesc, SynTexTrainer, RandomZData

IMAGESIZE = 224
LR = 5e-5
BETA1 = 0.9
BETA2 = 0.999
BATCH = 1
TEST_BATCH = 1

def INReLU(x, name=None):
    x = InstanceNorm('inorm', x)
    return tf.nn.relu(x, name=name)

def INLReLU(x, name=None):
    x = InstanceNorm('inorm', x)
    return tf.nn.leaky_relu(x, alpha=0.2, name=name)

class AdaptiveSynTex(SynTexModelDesc):
    def __init__(self, args):
        super(AdaptiveSynTex, self).__init__(args)
        self._image_size = args.get("image_size", IMAGESIZE)
        self._lr = args.get("lr", LR)
        self._beta1 = args.get("beta1", BETA1)
        self._beta2 = args.get("beta2", BETA2)
        self._n_stage = args.get("n_stage", 1)
        self._n_block = args.get("n_block", 2)
        act = args.get("act", "sigmoid")
        if act == "sigmoid":
            self._act = tf.nn.sigmoid
        elif act == "tanh":
            self._act = tf.nn.tanh
        elif act == "identity":
            self._act = tf.identity
        else:
            raise ValueError("Invalid activation: " + str(type(act)))
        self._loss_scale = args.get("loss_scale", 0.)
        self._pre_act = args.get("pre_act", False)
        self._nn_upsample = not args.get("deconv_upsample", False)

    def inputs(self):
        return [tf.TensorSpec((None, self._image_size, self._image_size, 3), tf.float32, 'pre_image_input'),
            tf.TensorSpec((None, self._image_size, self._image_size, 3), tf.float32, 'image_target')]

    @staticmethod
    def get_parser(ps=None):
        ps = SynTexModelDesc.get_parser(ps)
        ps = ArgParser(ps, name="adaptive")
        ps.add("--image-size", type=int, default=IMAGESIZE)
        ps.add("--lr", type=float, default=LR)
        ps.add("--beta1", type=float, default=BETA1)
        ps.add("--beta2", type=float, default=BETA2)
        ps.add("--n-stage", type=int, default=1)
        ps.add("--n-block", type=int, default=2, help="number of res blocks in each scale.")
        ps.add("--act", type=str, default="sigmoid", choices=["sigmoid", "tanh", "identity"])
        ps.add("--loss-scale", type=float, default=0.)
        ps.add_flag("--pre-act")
        ps.add_flag("--deconv-upsample")
        return ps

    @staticmethod
    def build_res_block(x, name, chan, first=False):
        """The value of input is considered to be computed after normalization
        but before non-linearity. So the activation function is needed to be
        applied in the residual branch.

        This implementation assumes that the input and output dimensions are
        consistent. The upsampling and downsampling are implemented in other
        modules such as Conv2D_Transpose and Conv2d, or resize_nearest_neighbor
        and avg_pool.
        """
        with tf.variable_scope(name):
            assert x.get_shape().as_list()[-1] == chan
            shortcut = x
            res_input = tf.nn.relu(x, "act_input")
            return (LinearWrap(res_input)
                .tf.pad([[0, 0], [1, 1], [1, 1], [0, 0]], mode="SYMMETRIC")
                .Conv2D('conv0', chan, 3, padding="VALID")
                .tf.pad([[0, 0], [1, 1], [1, 1], [0, 0]], mode="SYMMETRIC")
                .Conv2D('conv1', chan, 3, padding="VALID", activation=tf.identity)
                .InstanceNorm('inorm')
            )() + shortcut

    @ staticmethod
    def build_pre_res_block(x, name, chan, first=False):
        """The value of input is considered to be computed before normalization
        and non-linearity. So the normalization and activation functions are
        needed to be applied in the residual branch.

        This implementation assumes that the input and output dimensions are
        consistent. The upsampling and downsampling are implemented in other
        modules such as Conv2D_Transpose and Conv2d, or resize_nearest_neighbor
        and avg_pool.
        """
        with tf.variable_scope(name):
            assert x.get_shape().as_list()[-1] == chan
            shortcut = x
            res_input = INReLU(x, "act_input")
            return (LinearWrap(res_input)
                .tf.pad([[0,0], [1,1], [1,1], [0,0]], mode="SYMMETRIC")
                .Conv2D("conv0", chan, 3, padding="VALID")
                .tf.pad([[0,0], [1,1], [1,1], [0,0]], mode="SYMMETRIC")
                .Conv2D("conv1", chan, 3, padding="VALID", activation=tf.identity)
            )() + shortcut

    @staticmethod
    def build_upsampling_nn(x, name, scale=2, chan=None, ksize=None):
        _, h, w, c = x.get_shape().as_list()
        new_size = [h*scale, w*scale]
        if chan is None:
            chan = c
        if ksize is None:
            ksize = scale*2 + 1
        p_start = ksize // 2
        p_end = ksize - 1 - p_start
        x_up = tf.image.resize_nearest_neighbor(x, size=new_size, name=name+"_nn")
        x_up = tf.pad(x_up, [[0,0], [p_start, p_end], [p_start, p_end], [0,0]], mode="SYMMETRIC") 
        return Conv2D(name, x_up, chan, ksize, padding="VALID", activation=tf.identity, use_bias=False)

    @staticmethod
    def build_upsampling_deconv(x, name, scale=2, chan=None, ksize=None):
        if chan is None:
            chan = x.get_shape().as_list()[-1]
        if ksize is None:
            ksize = scale*2
        return Conv2DTranspose(name, x, chan, ksize,
            strides=scale, activation=tf.identity, use_bias=False)

    def build_stage(self, pre_image_input, gram_target, name="stage"):
        res_block = AdaptiveSynTex.build_pre_res_block if self._pre_act \
            else AdaptiveSynTex.build_res_block
        upsample = AdaptiveSynTex.build_upsampling_nn if self._nn_upsample \
            else AdaptiveSynTex.build_upsampling_deconv
        with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
            # extract features and gradients
            image_input = self._act(pre_image_input, name="input_"+name)
            feat_input, _ = self._build_extractor(image_input, calc_gram=False)
            loss_overall_input, loss_layer_input, grad_layer = \
                build_texture_loss(feat_input, gram_target,
                    SynTexModelDesc.DEFAULT_COEFS, calc_grad=True, name="grad")
            # For an adaptive texture synthesizer, we provide gradients explicitly to
            # the synthesizer.
            #            none +
            # grad[4] conv[4] -> res[4] -> up[4] +
            #                    grad[3] conv[3] -> res[3] -> up[3] +
            #                                       grad[2] conv[2] -> res[2] -> up[2] +
            #                                                               ... ...
            #                                                 up[1] +
            #                                       grad[0] conv[0] -> res[0] -> output
            with argscope([Conv2D, Conv2DTranspose], activation=INReLU, use_bias=False):
                first = True
                for layer in reversed(feat_input):
                    grad = grad_layer[layer]
                    chan = grad.get_shape().as_list()[-1]
                    with tf.variable_scope(layer):
                        # compute pseudo grad of current layer
                        grad = Conv2D("grad_conv1", grad, chan, 3)
                        grad = Conv2D("grad_conv2", grad, chan, 3, activation=tf.identity)
                        # merge with grad from deeper layers
                        if first:
                            delta = tf.identity(grad, name="grad_merged")
                            first = False
                        else:
                            # upsample deeper grad
                            if self._pre_act:
                                delta = INReLU(delta, "pre_inrelu")
                            else:
                                delta = tf.nn.relu(delta, "pre_relu")
                            delta = upsample(delta, "up", chan=chan)
                            # add two grads
                            delta = tf.add(grad, delta, name="grad_merged")
                        if not self._pre_act:
                            delta = InstanceNorm("post_inorm", delta)
                        # simulate the backpropagation procedure to next level
                        for k in range(self._n_block):
                            delta = res_block(delta, "res{}".format(k), chan, first=(k == 0))
                # output
                if self._pre_act:
                    delta = INReLU(delta, "actlast")
                else:
                    delta = tf.nn.relu(delta, "actlast")
                delta = tf.pad(delta, [[0, 0], [1, 1], [1, 1], [0, 0]], mode="SYMMETRIC")
                delta_input = Conv2D("convlast", delta, 3, 3, padding="VALID", activation=tf.identity, use_bias=True)
            pre_image_output = tf.add(pre_image_input, delta_input, name="pre_image_output")
        return image_input, loss_overall_input, loss_layer_input, pre_image_output

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
                    self.build_stage(pre_image_output, gram_target, name="stage%d" % s)
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
        lr = tf.get_variable("learning_rate", initializer=self._lr, trainable=False)
        return tf.train.AdamOptimizer(lr, beta1=self._beta1, beta2=self._beta2, epsilon=1e-3)


def get_data(datadir, size=IMAGESIZE, isTrain=True):
    if isTrain:
        augs = [
            imgaug.RandomCrop(size),
            imgaug.Flip(horiz=True),
        ]
    else:
        augs = [imgaug.CenterCrop(size)]

    def get_images(dir):
        files = sorted(glob.glob(os.path.join(dir, "*.jpg")))
        df = ImageFromFile(files, channel=3, shuffle=isTrain)
        random_df = RandomZData([size, size, 3], -1, 1)
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
    ps = AdaptiveSynTex.get_parser()
    ps.add("--data-folder", type=str, default="../images/single_12")
    ps.add("--save-folder", type=str, default="train_log/single_model")
    ps.add("--steps-per-epoch", type=int, default=1000)
    ps.add("--max-epoch", type=int, default=200)
    ps.add("--save-epoch", type=int, default=20)
    ps.add("--image-steps", type=int, default=100)
    args = ps.parse_args()
    print("Arguments")
    ps.print_args()
    print()

    data_folder = args.get("data_folder", "../images/single_12")
    save_folder = args.get("save_folder", "train_log/single_model")
    image_size = args.get("image_size", IMAGESIZE)
    steps_per_epoch = args.get("steps_per_epoch", 1000)
    max_epoch = args.get("max_epoch", 200)
    save_epoch = args.get("save_epoch", max_epoch // 10)
    image_steps = args.get("image_steps", 100)
    lr = args.get("lr", LR)
    # lr starts decreasing at half of max epoch
    start_dec_epoch = max_epoch // 2
    # stop when lr is 0.01 of its initial value
    end_epoch = max_epoch - int((max_epoch - start_dec_epoch) * 0.01)

    if save_folder == None:
        logger.auto_set_dir()
    else:
        logger.set_logger_dir(save_folder)

    df = get_data(data_folder, image_size)
    df = PrintData(df)
    data = QueueInput(df)

    SynTexTrainer(data, AdaptiveSynTex(args)).train_with_defaults(
        callbacks=[
            PeriodicTrigger(ModelSaver(), every_k_epochs=save_epoch),
            PeriodicTrigger(ModelSaver(), every_k_epochs=end_epoch), # save model at last
            ScheduledHyperParamSetter(
                'learning_rate',
                [(start_dec_epoch, lr), (max_epoch, 0)], interp="linear"),
            #PeriodicTrigger(VisualizeTestSet(data_folder, image_size), every_k_epochs=10),
            MergeAllSummaries(period=10), # scalar only
            MergeAllSummaries(period=image_steps, key="image_summaries"),
        ],
        max_epoch= end_epoch,
        steps_per_epoch=steps_per_epoch,
        session_init=None
    )
