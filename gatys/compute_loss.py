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
        ps.add("--image-size", type=int, default=224)
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
        ph_target = tf.placeholder(tf.float32, shape=self.shape_input, name="targets")
        ph_is_preimage = tf.placeholder(tf.bool, shape=[], name="is_preimage")
        image_input = tf.cond(ph_is_preimage, lambda: tf.sigmoid(ph_input), lambda: ph_input, name="image_input")
        image_target = tf.cond(ph_is_preimage, lambda: tf.sigmoid(ph_target), lambda: ph_target, name="image_target")
        #
        feat_input = vgg19.build(image_input, "ext_input")
        feat_target = vgg19.build(image_target, "ext_target")
        #
        loss_overall, loss_layer, _ = \
            build_texture_loss(feat_input, feat_target, Synthesizer.DEFAULT_COEFS, is_gram_b=False)
        #
        self.ph_input = ph_input
        self.ph_target = ph_target
        self.ph_is_preimage = ph_is_preimage
        self.loss_overall = loss_overall
        self.step_fetch_dict = {"func": self.loss_overall, "feat": feat_input["pool4"]}

    def step(self, input, target_image, is_preimage=False):
        """First call 'compute_gram' with 'update=True' to load target gram
        """
        if input.ndim == 3:
            input = input[np.newaxis, ...]
        if target_image.ndim == 3:
            target = target_image[np.newaxis, ...]
        return self.sess.run(self.step_fetch_dict,
            {self.ph_input: input, self.ph_target: target, self.ph_is_preimage: is_preimage})

    def compute_loss(self, input_init, image_target, is_preimage=False):
        ret = self.step(input_init, image_target, is_preimage=is_preimage)
        return ret["func"], ret["feat"]

def resize_crop(img, size, crop_size=None, interp=cv2.INTER_LINEAR):
    # resize shortest edge and center crop
    assert img.ndim == 3
    h, w = img.shape[:2]
    if crop_size is None:
        crop_size = size
    if h < w:
        newh, neww = size, int(w * 1. * size/h + 0.5)
        h_off = 0
        w_off = (neww - crop_size) // 2
    else:
        newh, neww = int(h * 1. * size/w + 0.5), size
        h_off = (newh - crop_size) // 2
        w_off = 0
    ret = cv2.resize(img, (neww, newh), interpolation=interp)
    if img.ndim == 3 and ret.ndim == 2:
        ret = ret[:,:, np.newaxis]
    return ret[h_off:h_off+crop_size, w_off:w_off+crop_size, :]

def compute_loss():
    from imageio import imread
    import time
    import sys, os
    from natsort import natsorted

    from file_utils import get_image_file_names
    from syntex.visualization import ImageDisplay, get_plottable_data

    ps = Synthesizer.get_parser()
    ps.add("--image-path", type=str, default="../images/flower_beds_256/FlowerBeds0008_256.jpg")
    ps.add("--image-folder", type=str, default="../images/scaly/train")
    ps.add("--test-image-path", type=str, default="test_log")
    ps.add("--test-image-folder", type=str, default="test_log")
    ps.add("--save-folder", type=str, default="train_log")
    ps.add_flag("--batch-test")
    ps.add("--batch-test-start", type=int, default=0)
    ps.add("--batch-test-end", type=int, default=None)
    ps.add_flag("--preimage")
    ps.add("--resize", type=int, default=256)
    args = ps.parse_args()
    #
    is_preimage = False
    image_size = args.get("image_size")
    resize = args.get("resize")
    batch_test = args.get("batch_test")
    NUM_PARALLEL_EXEC_UNITS = 2
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
            test_image_folder = args.get("test_image_folder")
            test_image_names = get_image_file_names(test_image_folder)
            test_image_names = natsorted(test_image_names)
            save_folder = args.get("save_folder")
            batch_test_start = args.get("batch_test_start")
            batch_test_end = args.get("batch_test_end")
            losses = list()
            if not os.path.exists(save_folder):
                os.makedirs(save_folder)
            imdp = ImageDisplay()
            for name, test_name in list(zip(image_names, test_image_names))[batch_test_start:batch_test_end]:
                image_path = os.path.join(image_folder, name)
                image_target = imread(image_path, pilmode="RGB") / 255.
                image_target = resize_crop(image_target, resize, image_size)
                test_image_path = os.path.join(test_image_folder, test_name)
                test_image = imread(test_image_path, pilmode="RGB") / 255.
                if test_image.shape[1] > image_size:
                    image_target = test_image[:, -image_size:, :]
                    test_image = test_image[:, -image_size*2:-image_size, :]
                #test_image = resize_crop(test_image, image_size)
                #
                start_time = time.time()
                print(image_path, test_image_path)
                func, feat = syn.compute_loss(test_image, image_target)
                print("Finished (%.2f sec)" % (time.time() - start_time), func)
                losses.append(func)
            print("mean =", np.mean(losses))
        else:
            image_path = args.get("image_path")
            image_target = imread(image_path, pilmode="RGB") / 255.
            image_target = resize_crop(image_target, resize, image_size)
            test_image_path = args.get("test_image_path")
            test_image = imread(test_image_path, pilmode="RGB") / 255.
            if test_image.shape[1] > image_size:
                image_target = test_image[:, -image_size:, :]
                test_image = test_image[:, -image_size*2:-image_size, :]
            #
            start_time = time.time()
            print(image_path, test_image_path)
            func, feat = syn.compute_loss(test_image, image_target)
            print("Finished (%.2f sec)" % (time.time() - start_time), func)


if __name__ == "__main__":
    #test()
    compute_loss()