from collections import defaultdict
from io import IOBase

class LineReader(object):
    @staticmethod
    def read_data(file_or_path, dtype, n_lines=None, skip=0, sep=' ', category_sep='|', is_first_name=True):
        if isinstance(file_or_path, str):
            fin = open(file_or_path, 'r')
        else:
            assert isinstance(file_or_path, IOBase)
            fin = file_or_path
        for _ in range(skip):
            fin.readline()
        data = defaultdict(list)
        if n_lines is None:
            for line in fin:
                line = line.rstrip('\r\n ')
                for i, cat_line in enumerate(line.split(category_sep)):
                    if cat_line:
                        items = cat_line.split(sep)
                        if is_first_name:
                            data[items[0]].append([dtype(item) for item in items[1:]])
                        else:
                            data["cat_%d" % i].append([dtype(item) for item in items])
        else:
            for i in range(n_lines):
                line = fin.readline()
                line = line.rstrip('\r\n ')
                for i, cat_line in enumerate(line.split(category_sep)):
                    if cat_line:
                        items = cat_line.split(sep)
                        if is_first_name:
                            data[items[0]].append([dtype(item) for item in items[1:]])
                        else:
                            data["cat_%d" % i].append([dtype(item) for item in items])
        return data


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import numpy as np
    path = "vlog.log"
    skip = 3
    n_lines = 100
    dtype = float
    eps = 1e-8
    layer = "conv1_1"

    data = LineReader.read_data(path, dtype, n_lines=n_lines, skip=skip)
    np_data = dict()
    for k in data:
        np_data[k] = np.array(data[k])

    # balance scale
    for k in np_data:
        mean = np.mean(np_data[k], axis=0)
        std = np.std(np_data[k], axis=0)
        #np_data[k] -= mean
        np_data[k] -= np.min(np_data[k], axis=0)
        np_data[k] /= (std + eps)
    
    fig = plt.figure(1, figsize=(10, 10))
    x = np.arange(np_data[layer].shape[0])
    act = np_data[layer][:, 0]
    gl = np_data[layer][:, 1]
    gi = np_data[layer][:, 2]
    lines = plt.plot(x, act, x, gl, x, gi)
    plt.axhline(y=0, color='k', linewidth=0.3, linestyle='-')
    plt.legend(("act", "grad layer", "grad input"))
    plt.title(layer)
    #fig.waitforbuttonpress()
    plt.savefig(layer+".png")