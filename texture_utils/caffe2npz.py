import numpy as np
import os.path as osp
from argparse import ArgumentParser
import caffe

def save_dict(args, output):
    eval("np.savez(r'%s', %s)" % (args.output, ','.join(["%s=output['%s']" % (k,k) for k in output.keys()])))

def main(args):
    net = caffe.Classifier(args.net_model, args.net_param)
    params = net.params
    output = dict()
    for k in params.keys():
        p = params[k]
        if len(p) == 2:
            w = np.transpose(np.array(p[0].data).astype(np.float32), [2,3,1,0])
            b = np.array(p[1].data).astype(np.float32)
            output[k] = [w,b]
        else:
            print("Skip layer", k, "with", len(p), "param blobs")
    save_dict(args, output)
        
def get_parser():
    ps = ArgumentParser()
    ps.add_argument('--net-model', type=str, default='Models/VGG19_ave_pool_deploy.prototxt')
    ps.add_argument('--net-param', type=str, default='Models/vgg19_normalised.caffemodel')
    ps.add_argument('--output', type=str, default='vgg19_normalised.npz')
    return ps
    
if __name__ == '__main__':
    ps = get_parser()
    args = ps.parse_args()
    main(args)