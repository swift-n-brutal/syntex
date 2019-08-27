import tensorflow as tf
import numpy as np
from collections import OrderedDict
from scipy.optimize import minimize

from aparse import ArgParser
from syntex.texture_utils import Vgg19Extractor, build_gram, build_texture_loss

def get_bounds(shape=None):
    bounds = list()
    for i in range(np.prod(shape)):
        bounds.append([0., 1.])
    return bounds

class Synthesizer(object):
    DEFAULT_COEFS = OrderedDict([
        ("conv1_1", 1e5),
        ("pool1", 1e5),
        ("pool2", 1e5),
        ("pool3", 1e5),
        ("pool4", 1e5)
    ])

    def __init__(self, args, sess):
        self.vgg19_path = args.get("vgg19_npy_path")
        self.image_size = args.get("image_size")
        self.gram_only = args.get("gram_only")
        self.options = {
            "maxiter": args["maxiter"],
            "maxcor": args["maxcor"],
            "disp": args["disp"],
            "ftol": 0.,
            "gtol": 0.
        }
        self.sess = sess
    
    @staticmethod
    def get_parser(ps=None):
        ps = ArgParser(ps, name="synthesizer")
        ps.add("--vgg19-npy-path")
        ps.add("--image-size", type=int, default=256)
        ps.add_flag("--gram-only")
        #
        ps.add("--maxiter", type=int, default=1000,
                help="Max iterations of optimizer")
        ps.add("--maxcor", type=int, default=20,
                help="Max number of history gradients and points")
        ps.add_flag("--disp")
        return ps

    def build(self):
        vgg19 = Vgg19Extractor(self.vgg19_path)
        self.shape_input = [1, self.image_size, self.image_size, 3]
        ph_input = tf.placeholder(tf.float32, shape=self.shape_input, name="inputs")
        #
        feat_input = vgg19.build(ph_input, "ext_input")
        gram_input = OrderedDict()
        for k in Synthesizer.DEFAULT_COEFS:
            gram_input[k] = build_gram(feat_input[k])
        #
        with tf.variable_scope("buf"):
            if not self.gram_only:
                gram_target = OrderedDict()
                op_update_gram_target = list()
                for k in Synthesizer.DEFAULT_COEFS:
                    gram_target[k] = tf.get_variable("gram_"+k,
                            shape=gram_input[k].shape.as_list(),
                            dtype=tf.float32, trainable=False)
                    op_update_gram_target.append(tf.assign(gram_target[k], gram_input[k]))
        #
        if not self.gram_only:
            loss_overall, loss_layer, _ = build_texture_loss(feat_input, gram_target, Synthesizer.DEFAULT_COEFS)
            grad_input = tf.gradients(loss_overall, ph_input, name="grad_input")[0]
        #
        self.ph_input = ph_input
        self.gram_input = gram_input
        if not self.gram_only:
            self.op_update_gram_target = op_update_gram_target
            self.loss_overall = loss_overall
            self.grad_input = grad_input

    def compute_gram(self, image, mask=None, update=False):
        if len(image.shape) == 3:
            image = image[np.newaxis, ...]
        if update:
            return self.sess.run(self.op_update_gram_target, feed_dict={self.ph_input: image})
        else:
            return self.sess.run(self.gram_input, feed_dict={self.ph_input: image})

    def step(self, image, mask=None, target_image=None):
        """First call 'compute_gram' with 'update=True' to load target gram
        """
        if len(image.shape) == 3:
            image = image[np.newaxis, ...]
        fetch_dict = {"grad": self.grad_input, "func": self.loss_overall}
        return self.sess.run(fetch_dict, {self.ph_input: image})

    def synthesize(self, image_init, image_target, mask_input=None, mask_target=None, bounds=None):
        def f(x):
            x = x.reshape(self.shape_input)
            ret = self.step(x)
            return [np.mean(ret["func"]), np.array(ret["grad"].ravel(), dtype=np.float64)]
        
        if bounds is None:
            bounds = get_bounds(self.shape_input)
        self.compute_gram(image_target, update=True)
        res = minimize(f, image_init, method="L-BFGS-B", jac=True,
                bounds=bounds, options=self.options)
        return res["x"].reshape(self.shape_input[1:]), res["fun"]


def test():
    from imageio import imread
    import time

    from syntex.visualization import ImageDisplay, get_plottable_data

    ps = Synthesizer.get_parser()
    args = ps.parse_args()
    #
    image_target = imread("../images/flower_beds_256/FlowerBeds0008_256.jpg", pilmode="RGB") / 255.
    image_init = np.random.uniform(1./255, 1. - 1./255, size=image_target.shape)
    #
    start_time = time.time()
    config = tf.ConfigProto(log_device_placement=False, device_count={"GPU": 0})
    with tf.Session(config=config) as sess:
        syn = Synthesizer(args, sess)
        syn.build()
        image_syn, fun = syn.synthesize(image_init, image_target)
    print("Finished (%.2f sec)" % (time.time() - start_time))
    #
    imdp = ImageDisplay()
    imdp.show_images([
        (get_plottable_data(image_target, scale=255.), "Original"),
        (get_plottable_data(image_syn, scale=255.), "%.4e" % fun)])

if __name__ == "__main__":
    test()