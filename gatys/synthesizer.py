import tensorflow as tf
import numpy as np
from collections import OrderedDict, defaultdict
from scipy.optimize import minimize
import cv2

from aparse import ArgParser
from text_utils import print_dict
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
        self.verbose = args.get("verbose")
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
        ps.add_flag("--verbose")
        return ps

    def build(self):
        vgg19 = Vgg19Extractor(self.vgg19_path)
        self.shape_input = [1, self.image_size, self.image_size, 3]
        ph_input = tf.placeholder(tf.float32, shape=self.shape_input, name="inputs")
        ph_is_preimage = tf.placeholder(tf.bool, shape=[], name="is_preimage")
        image_input = tf.cond(ph_is_preimage, lambda: tf.sigmoid(ph_input), lambda: ph_input, name="image_input")
        #
        feat_input = vgg19.build(image_input, "ext_input")
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
            loss_overall, loss_layer, grad_layer = build_texture_loss(feat_input, gram_target, Synthesizer.DEFAULT_COEFS,
                calc_grad=self.verbose)
            grad_input = tf.gradients(loss_overall, ph_input, name="grad_input")[0]
            if self.verbose:
                # compute the mean absolute value of gradients from each layer
                L1_act = OrderedDict()
                L1_grad_layer = OrderedDict()
                L1_grad_input = OrderedDict()
                L1_grad_input["all"] = tf.reduce_mean(tf.abs(grad_input))
                for k in Synthesizer.DEFAULT_COEFS:
                    L1_act[k] = tf.reduce_mean(tf.abs(feat_input[k]))
                    L1_grad_layer[k] = tf.reduce_mean(tf.abs(grad_layer[k]))
                    L1_grad_input[k] = tf.reduce_mean(tf.abs(tf.gradients(loss_layer[k], ph_input)[0]))
        #
        self.ph_input = ph_input
        self.ph_is_preimage = ph_is_preimage
        self.gram_input = gram_input
        if not self.gram_only:
            self.op_update_gram_target = op_update_gram_target
            self.loss_overall = loss_overall
            self.grad_input = grad_input
        self.step_fetch_dict = {"grad": self.grad_input, "func": self.loss_overall}
        if self.verbose:
            self.verbose_fetch_dict = {
                "func": tf.reduce_sum(self.loss_overall),
                "gi_all": L1_grad_input['all']}
            for k in Synthesizer.DEFAULT_COEFS:
                self.verbose_fetch_dict["l_"+k] = tf.reduce_sum(loss_layer[k])
                self.verbose_fetch_dict["a_"+k] = L1_act[k]
                self.verbose_fetch_dict["gl_"+k] = L1_grad_layer[k]
                self.verbose_fetch_dict["gi_"+k] = L1_grad_input[k]

    def compute_gram(self, image, mask=None, update=False, is_preimage=False):
        if len(image.shape) == 3:
            image = image[np.newaxis, ...]
        if update:
            return self.sess.run(self.op_update_gram_target,
                feed_dict={self.ph_input: image, self.ph_is_preimage: is_preimage})
        else:
            return self.sess.run(self.gram_input,
                feed_dict={self.ph_input: image, self.ph_is_preimage: is_preimage})

    def step(self, input, mask=None, target_image=None, is_preimage=False):
        """First call 'compute_gram' with 'update=True' to load target gram
        """
        if len(input.shape) == 3:
            input = input[np.newaxis, ...]
        return self.sess.run(self.step_fetch_dict,
            {self.ph_input: input, self.ph_is_preimage: is_preimage})

    def synthesize(self, input_init, image_target, mask_input=None, mask_target=None, bounds=None, is_preimage=False):
        def f(x):
            x = x.reshape(self.shape_input)
            ret = self.step(x, is_preimage=is_preimage)
            # We want the magnitude of grad to be independent of the batchsize
            return [np.sum(ret["func"]), np.array(ret["grad"].ravel(), dtype=np.float64)]

        def verbose_callback(x, *args, **kwargs):
            x = x.reshape(self.shape_input)
            ret = self.sess.run(self.verbose_fetch_dict,
                {self.ph_input: x, self.ph_is_preimage: is_preimage})
            for k in ret:
                self.verbose_dict[k].append(ret[k])
        
        if self.verbose:
            self.reset_verbose()
            cb = verbose_callback
        else:
            cb = None
        if bounds is None and (not is_preimage):
            bounds = get_bounds(self.shape_input)
        self.compute_gram(image_target, update=True)
        res = minimize(f, input_init, method="L-BFGS-B", jac=True,
                bounds=bounds, options=self.options, callback=cb)
        return res["x"].reshape(self.shape_input[1:]), res["fun"]

    def reset_verbose(self):
        if self.verbose:
            self.verbose_dict = defaultdict(list)

    def save_verbose(self, save_path):
        if self.verbose:
            np.savez(save_path, **self.verbose_dict)

def resize_crop(img, size, interp=cv2.INTER_LINEAR):
    # resize shortest edge and center crop
    assert img.ndim == 3
    h, w = img.shape[:2]
    if h < w:
        newh, neww = size, int(w * 1. * size/h + 0.5)
        h_off = 0
        w_off = (neww - size) // 2
    else:
        newh, neww = int(h * 1. * size/w + 0.5), size
        h_off = (newh - size) // 2
        w_off = 0
    ret = cv2.resize(img, (neww, newh), interpolation=interp)
    if img.ndim == 3 and ret.ndim == 2:
        ret = ret[:,:, np.newaxis]
    return ret[h_off:h_off+size, w_off:w_off+size, :]

def test_single(syn, image_target, is_preimage):
    if is_preimage:
        input_init = np.random.uniform(-1., 1., size=image_target.shape)
    else:
        input_init = np.random.uniform(1./255, 1. - 1./255, size=image_target.shape)
    output, func = syn.synthesize(input_init, image_target, is_preimage=is_preimage)
    if is_preimage:
        image_syn = 1./(1.+np.exp(-output))
    else:
        image_syn = output
    return image_syn, func
    
def test():
    from imageio import imread
    import time
    import sys, os

    from file_utils import get_image_file_names
    from syntex.visualization import ImageDisplay, get_plottable_data

    ps = Synthesizer.get_parser()
    ps.add("--image-path", type=str, default="../images/flower_beds_256/FlowerBeds0008_256.jpg")
    ps.add("--image-folder", type=str, default="../images/scaly/train")
    ps.add("--save-folder", type=str, default="train_log")
    ps.add_flag("--batch-test")
    ps.add_flag("--preimage")
    args = ps.parse_args()
    #
    is_preimage = args.get("preimage", False)
    image_size = args.get("image_size")
    batch_test = args.get("batch_test")
    NUM_PARALLEL_EXEC_UNITS = 8
    config = tf.ConfigProto(log_device_placement=False,
        intra_op_parallelism_threads=2,
        inter_op_parallelism_threads=2,
        allow_soft_placement=True,
        device_count={"GPU": 0, "CPU": NUM_PARALLEL_EXEC_UNITS})
    with tf.Session(config=config) as sess:
        # suggested by intel
        # https://software.intel.com/en-us/articles/tips-to-improve-performance-for-popular-deep-learning-frameworks-on-multi-core-cpus
        # https://blog.csdn.net/rockingdingo/article/details/55652662
        os.environ["OMP_NUM_THREADS"] = "%d" % NUM_PARALLEL_EXEC_UNITS
        os.environ["KMP_BLOCKTIME"] = "30"
        os.environ["KMP_SETTINGS"] = "1"
        os.environ["KMP_AFFINITY"]= "granularity=fine,verbose,compact,1,0"
        #
        syn = Synthesizer(args, sess)
        syn.build()
        if batch_test:
            image_folder = args.get("image_folder")
            image_names = get_image_file_names(image_folder)
            save_folder = args.get("save_folder")
            if not os.path.exists(save_folder):
                os.makedirs(save_folder)
            imdp = ImageDisplay()
            for name in image_names:
                image_path = os.path.join(image_folder, name)
                image_target = imread(image_path, pilmode="RGB") / 255.
                image_target = resize_crop(image_target, image_size)
                #
                start_time = time.time()
                print(image_path)
                image_syn, func = test_single(syn, image_target, is_preimage)
                print("Finished (%.2f sec)" % (time.time() - start_time))
                #
                save_path = os.path.join(save_folder, name.rsplit('.', 1)[0]+".npz")
                syn.save_verbose(save_path)
                #
                imdp.show_images([
                    (get_plottable_data(image_target, scale=255.), "Original"),
                    (get_plottable_data(image_syn, scale=255.), "%.4e" % func)],
                    wait=False, fig_id=1)
                save_path = os.path.join(save_folder, name)
                imdp.savefig(save_path)
                imdp.clear()
        else:
            image_path = args.get("image_path")
            image_target = imread(image_path, pilmode="RGB") / 255.
            image_target = resize_crop(image_target, image_size)
            #
            start_time = time.time()
            print(image_path)
            image_syn, func = test_single(syn, image_target, is_preimage)
            print("Finished (%.2f sec)" % (time.time() - start_time))
            #
            imdp = ImageDisplay()
            imdp.show_images([
                (get_plottable_data(image_target, scale=255.), "Original"),
                (get_plottable_data(image_syn, scale=255.), "%.4e" % func)])

if __name__ == "__main__":
    test()