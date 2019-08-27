import tensorflow as tf
import numpy as np
from imageio import imread
import time

from ffsyn.texture import Synthesizer
from ffsyn.visualization import ImageDisplay, get_plottable_data

def test():
    ps = Synthesizer.get_parser()
    args = ps.parse_args()
    args = vars(args)
    #
    image_target = imread("E:/projects/python/ffsyn/images/flower_beds_256/FlowerBeds0008_256.jpg", pilmode="RGB") / 255.
    image_init = np.random.uniform(1./255, 1. - 1./255, size=image_target.shape)
    #
    start_time = time.time()
    config = tf.ConfigProto(log_device_placement=False, device_count={"GPU": 0})
    with tf.Session(config=config) as sess:
        syn = Synthesizer(args, sess)
        syn.build()
        #image_syn = image_init
        image_syn, fun = syn.synthesize(image_init, image_target)
    print("Finished (%.2f sec)" % (time.time() - start_time))
    #
    imdp = ImageDisplay()
    imdp.show_images([
        (get_plottable_data(image_target, scale=255.), "Original"),
        (get_plottable_data(image_syn, scale=255.), "%.4e" % fun)])

if __name__ == "__main__":
    test()