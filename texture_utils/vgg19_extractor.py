from .vgg19 import Vgg19, VGG_MEAN, VGG_MEAN_RGB

class Vgg19Extractor(Vgg19):
    def __init__(self, vgg19_path=None, trainable=False, padding="SAME",
            is_rgb_input=True, use_avg_pool=True, topmost="pool4", name="vgg19_extractor"):
        super(Vgg19Extractor, self).__init__(vgg19_path, trainable, padding, is_rgb_input, use_avg_pool, topmost, name)
