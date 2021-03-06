import tensorflow as tf
import numpy as np
from collections import OrderedDict

from tensorpack import DataFlow, ModelDescBase, StagingInput, TowerTrainer
from tensorpack.graph_builder import DataParallelBuilder, LeastLoadedDeviceSetter
from tensorpack.tfutils.tower import TowerContext, TowerFuncWrapper
from tensorpack.utils.argtools import memoized_method

from syntex.texture_utils import Vgg19Extractor, build_gram, build_texture_loss
from syntex.aparse import ArgParser

class SynTexModelDesc(ModelDescBase):
    DEFAULT_COEFS = OrderedDict([
        ("conv1_1", 1e6),
        ("pool1", 1e6),
        ("pool2", 1e6),
        ("pool3", 2e6),
        ("pool4", 4e6)
    ])
    def __init__(self, args):
        self._vgg19 = Vgg19Extractor(args.get("vgg19_path"))
        self.loss = None
    
    @staticmethod
    def get_parser(ap=None):
        ap = ArgParser(ap, name="syntex-model")
        ap.add("--vgg19-path")
        return ap

    def collect_variables(self, scope=None):
        """
        Assign 'self.vars' to the trainable parameters under 'scope' if not None.
        """
        self.vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)
        assert self.vars

    def _build_loss(self, image_output, gram_target, calc_grad=False, name="texture_loss"):
        with tf.name_scope(name):
            feat_output = self._vgg19.build(image_output)
            loss_overall, loss_layer, calc_grad = build_texture_loss(
                feat_output, gram_target, SynTexModelDesc.DEFAULT_COEFS,
                calc_grad=calc_grad, name="loss"
            )
        return loss_overall, loss_layer, calc_grad

    def _build_extractor(self, image_input, calc_gram=True, name="ext"):
        with tf.name_scope(name):
            feat_input = self._vgg19.build(image_input)
            if calc_gram:
                gram_input = OrderedDict()
                for k in SynTexModelDesc.DEFAULT_COEFS:
                    gram_input[k] = build_gram(feat_input[k])
            else:
                gram_input = None
        return feat_input, gram_input

    def build_graph(self, *inputs):
        pass

    def optimizer(self):
        pass
    
    @memoized_method
    def get_optimizer(self):
        return self.optimizer()


class SynTexTrainer(TowerTrainer):
    def __init__(self, input, model, num_gpu=1):
        """
        Parameters
        ----------
        input: InputSource
        model: SynTexModelDesc
        """
        super(SynTexTrainer, self).__init__()
        assert isinstance(model, SynTexModelDesc), model

        if num_gpu > 1:
            input = StagingInput(input)

        # setup input
        cbs = input.setup(model.get_input_signature())
        self.register_callback(cbs)

        if num_gpu <= 1:
            self._build_trainer(input, model)
        else:
            self._build_multigpu_trainer(input, model, num_gpu)

    def _build_trainer(self, input, model):
        # build the graph
        self.tower_func = TowerFuncWrapper(model.build_graph, model.inputs())
        with TowerContext('', is_training=True):
            self.tower_func(*input.get_input_tensors())

        # define the training iteration
        with tf.name_scope("optimize"):
            opt = model.get_optimizer()
            op_min = opt.minimize(model.loss, var_list=model.vars,
                colocate_gradients_with_ops=True, name="op_min")
        self.train_op = op_min

    def _build_multigpu_trainer(self, input, model, num_gpu):
        assert num_gpu > 1, num_gpu
        raw_devices = ["/gpu:{}".format(k) for k in range(num_gpu)]

        # build the graph with multi-gpu replications
        def get_cost(*inputs):
            model.build_graph(*inputs)
            return model.loss

        self.tower_func = TowerFuncWrapper(get_cost, model.get_input_signature())
        devices = [LeastLoadedDeviceSetter(d, raw_devices) for d in raw_devices]
        cost_list = DataParallelBuilder.build_on_towers(
            list(range(num_gpu)),
            lambda: self.tower_func(*input.get_input_tensors()),
            devices
        )
        #
        with tf.name_scope("optimize"):
            loss = tf.add_n(cost_list) * (1.0 / num_gpu)
            opt = model.get_optimizer()
            op_min = opt.minimize(loss, var_list=model.vars,
                colocate_gradients_with_op=True, name="op_min")
        self.train_op = op_min

class RandomZData(DataFlow):
    def __init__(self, shape, minv=0., maxv=1.):
        super(RandomZData, self).__init__()
        self.shape = shape
        self.minv = minv
        self.maxv = maxv

    def __iter__(self):
        while True:
            yield [np.random.uniform(self.minv, self.maxv, size=self.shape)]
