import tensorflow as tf
import numpy as np
import glob, os

from tensorpack import (InstanceNorm, LinearWrap, Conv2D, Conv2DTranspose,
    argscope, imgaug, logger, PrintData, QueueInput, ModelSaver, Callback,
    ScheduledHyperParamSetter, PeriodicTrigger, SaverRestore, JoinData,
    AugmentImageComponent, ImageFromFile, BatchData, MultiProcessRunner,
    MergeAllSummaries)
from tensorpack.tfutils.summary import add_moving_summary

from aparse import ArgParser
from texture import build_texture_loss
from model import SynTexModelDesc, SynTexTrainer, RandomZData

IMAGESIZE = 256
LR = 2e-4
BETA1 = 0.5
BETA2 = 0.999
BATCH = 1
TEST_BATCH = 32

def INReLU(x, name=None):
    x = InstanceNorm('inorm', x)
    return tf.nn.relu(x, name=name)

def INLReLU(x, name=None):
    x = InstanceNorm('inorm', x)
    return tf.nn.leaky_relu(x, alpha=0.2, name=name)

class SingleSynTex(SynTexModelDesc):
    def __init__(self, args):
        super(SingleSynTex, self).__init__(args)
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

    def inputs(self):
        return [tf.TensorSpec((None, self._image_size, self._image_size, 3), tf.float32, 'pre_image_input'),
            tf.TensorSpec((None, self._image_size, self._image_size, 3), tf.float32, 'image_target')]

    @staticmethod
    def get_parser(ps=None):
        ps = ArgParser(ps, name="single")
        ps.add("--image-size", type=int, default=IMAGESIZE)
        ps.add("--lr", type=float, default=LR)
        ps.add("--beta1", type=float, default=BETA1)
        ps.add("--beta2", type=float, default=BETA2)
        ps.add("--n-stage", type=int, default=1)
        ps.add("--n-block", type=int, default=2, help="number of res blocks in each scale.")
        ps.add("--act", type=str, default="sigmoid", choices=["sigmoid", "tanh", "identity"])
        ps.add("--loss-scale", type=float, default=0.)
        return ps

    @staticmethod
    def build_res_block(x, name, chan, first=False):
        with tf.variable_scope(name):
            input = x
            return (LinearWrap(x)
                .tf.pad([[0, 0], [1, 1], [1, 1], [0, 0]], mode="SYMMETRIC")
                .Conv2D('conv0', chan, 3, padding="VALID")
                .tf.pad([[0, 0], [1, 1], [1, 1], [0, 0]], mode="SYMMETRIC")
                .Conv2D('conv1', chan, 3, padding="VALID", activation=tf.identity)
                .InstanceNorm('inorm')()) + input

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
            tf.summary.image(name, im, max_outputs=10)

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
        # average losses from all stages
        weights = [1.]
        for i in range(len(self.losses) - 1):
            weights.append(weights[-1] * self._loss_scale)
        # skip the first loss as it is computed from noise
        self.loss = tf.add_n([weights[i] * loss \
            for i, loss in enumerate(reversed(self.losses[1:]))], name="loss")
        # summary
        viz("stages-target", self.image_outputs + [image_target])
        add_moving_summary(self.loss, *self.losses)

    def build_stage(self, pre_image_input, gram_target, name="stage"):
        with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
            image_input = self._act(pre_image_input, name="input_"+name)
            feat_input, _ = self._build_extractor(image_input, calc_gram=False)
            loss_overall_input, loss_layer_input, _ = \
                build_texture_loss(feat_input, gram_target,
                    SynTexModelDesc.DEFAULT_COEFS, calc_grad=False, name="grad")
            # For a single texture synthesizer, we want to make the network
            # remember the characteristics of the texture. Thus, we do not
            # provide the grad information to the synthisizer through any
            # intermediate layers, but only through the final loss.
            # f[4] -> res[4] -> up[4] +
            #                    f[3] -> res[3] -> up[3] +
            #                                       f[2] -> res[2] -> up[2] +
            #                                                          f[1] -> res[1] -> up[1] +
            #                                                                             f[0] -> res[0] -> output
            with argscope([Conv2D, Conv2DTranspose], activation=INReLU):
                first = True
                for layer in reversed(feat_input):
                    feat = feat_input[layer]
                    chan = feat.get_shape().as_list()[-1]
                    if first:
                        l = tf.identity(feat, name=layer+"_iden")
                        first = False
                    else:
                        l = Conv2DTranspose(layer+"_deconv", l, chan, 3, strides=2, activation=tf.identity)
                        l = tf.add(feat, l, name=layer+"_add")
                        l = InstanceNorm(layer+"_inorm", l)
                    for k in range(self._n_block):
                        l = SingleSynTex.build_res_block(l, layer+"_res{}".format(k), chan, first=(k == 0))
                # output
                l = tf.pad(l, [[0, 0], [1, 1], [1, 1], [0, 0]])
                delta_input = Conv2D("convlast", l, 3, 3, padding="VALID", activation=tf.identity, use_bias=True)
            pre_image_output = tf.add(pre_image_input, delta_input, name="pre_image_output")
        return image_input, loss_overall_input, loss_layer_input, pre_image_output

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
        random_df = RandomZData([IMAGESIZE, IMAGESIZE, 3], -1, 1)
        return JoinData([random_df, AugmentImageComponent(df, augs)])
    
    names = ['train']  if isTrain else ['test']
    df = get_images(*[os.path.join(datadir, n) for n in names])
    df = BatchData(df, BATCH if isTrain else TEST_BATCH)
    return df


class VisualizeTestSet(Callback):
    def _setup_graph(self):
        self.pred = self.trainer.get_predictor(
            ["pre_image_input", "image_target"], ['stages-target/viz'])

    def _before_train(self):
        global data_folder
        self.val_ds = get_data(data_folder, isTrain=False)
        self.val_ds.reset_state()

    def _trigger(self):
        idx = 0
        for pii, it in self.val_ds:
            viz = self.pred(pii, it)
            self.trainer.monitors.put_image('test-{}'.format(idx), viz)
            idx += 1


if __name__ == "__main__":
    data_folder = "images/single_12"

    logger.auto_set_dir()

    df = get_data(data_folder)
    df = PrintData(df)
    data = QueueInput(df)

    SynTexTrainer(data, SingleSynTex(dict())).train_with_defaults(
        callbacks=[
            ModelSaver(),
            ScheduledHyperParamSetter(
                'learning_rate',
                [(100, 2e-4), (200, 0)], interp="linear"),
            #PeriodicTrigger(VisualizeTestSet(), every_k_epochs=10),
            MergeAllSummaries(period=10),
        ],
        max_epoch= 195,
        steps_per_epoch=1000,
        session_init=None
    )