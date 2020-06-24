import numpy as np

from .image_display import ImageDisplay

def get_plottable_data(data, scale=None, offset=None, minval=0, maxval=255, dtype=np.uint8):
    if scale is not None:
        data = data * scale
    if offset is not None:
        data = data + offset
    data = np.clip(data, minval, maxval).astype(dtype)
    return data