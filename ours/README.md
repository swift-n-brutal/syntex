

# Pseudo Optimizer
We reduce the optimization process into a prediction problem by training a feed-forward network that maps the per-layer gradients of the objective function to the optimal solution.

![fig-po](../figures/fig-po.png)


## Teaser

Interpolation among three random noise inputs.

|Interpolation|Target|Interpoation|Target|
|----         |----  |----        |----  |
|![interp1][interp1]|![target1][target1]|![interp2][interp2]|![target2][target2]|
|![interp3][interp3]|![target3][target3]|![interp4][interp4]|![target4][target4]|
|![interp5][interp5]|![target5][target5]|![interp6][interp6]|![target6][target6]|
|![interp7][interp7]|![target7][target7]|![interp8][interp8]|![target8][target8]|
|![interp9][interp9]|![target9][target9]|![interp10][interp10]|![target10][target10]|

## Usage

### Prepare VGG model
Download VGG model and convert it to an npz file following the steps in [../texture_utils](../texture_utils).

### Train
Prepare your training images as follows:
```
DATA_FOLDER/
|-- train/
    |-- image_123.jpg
    |-- image_456.jpg
|-- test/
    |-- image_789.jpg
```


Run the following command to train a model and save the logs and checkpoints to `SAVE_FOLDER`
```
CUDA_VISIBLE_DEVICES=0 python improved_model.py --data-folder DATA_FOLDER --save-folder SAVE_FOLDER
```

For the details of more arguments, run the following command
```
python xxx_model.py --help
```

### Test

Run the following command to test a pretrained model from the checkpoint `PRETRAINED_MODEL_CKPT` and save the synthesized images to `TEST_FOLDER`
```
CUDA_VISIBLE_DEVICES=0 python improved_model.py --test-only --data-folder DATA_FOLDER --test-ckpt PRETRAINED_MODEL_CKPT --test-folder TEST_FOLDER
```
A checkpoint file is named like `train_log/folder/model-ITERATION`.

### Models

* `improved_model.py` The latest model of the ProPO architecture with improved performance. Use this model to reproduce the results. The other models may not be up-to-date and may cause import errors.
* `progressive_model.py` The model of the ProPO architecture without tuning.
* `adaptive_model.py` The model of the AdaPO architecture.
* `model.py` The base _ModelDesc_ class used by tensorpack.


[//]: <links>
[interp1]:https://wx1.sinaimg.cn/large/006tWCFjly1get5q8du8hg3068068x6s.gif
[target1]:https://wx2.sinaimg.cn/large/006tWCFjly1get7dv1b47j3068068wfm.jpg
[interp2]:https://wx1.sinaimg.cn/large/006tWCFjly1get5r1l225g3068068x6s.gif
[target2]:https://wx3.sinaimg.cn/large/006tWCFjly1get7dwt3fvj3068068jsi.jpg
[interp3]:https://wx3.sinaimg.cn/large/006tWCFjly1get5rk5a7pg3068068qv8.gif
[target3]:https://wx4.sinaimg.cn/large/006tWCFjly1get7dyxinzj3068068gm0.jpg
[interp4]:https://wx1.sinaimg.cn/large/006tWCFjly1get5ru4d3hg3068068x6s.gif
[target4]:https://wx2.sinaimg.cn/large/006tWCFjly1get7dzgu08j3068068gmm.jpg
[interp5]:https://wx2.sinaimg.cn/large/006tWCFjly1get6xne2jbg3068068x6s.gif
[target5]:https://wx3.sinaimg.cn/large/006tWCFjly1get7e12f21j30680683zk.jpg
[interp6]:https://wx3.sinaimg.cn/large/006tWCFjly1get6xzar7vg3068068x6s.gif
[target6]:https://wx3.sinaimg.cn/large/006tWCFjly1get7e3nunoj3068068t9o.jpg
[interp7]:https://wx4.sinaimg.cn/large/006tWCFjly1get6yc0vw3g3068068x6s.gif
[target7]:https://wx2.sinaimg.cn/large/006tWCFjly1get7e8k94mj3068068dgs.jpg
[interp8]:https://wx4.sinaimg.cn/large/006tWCFjly1get6ylgyblg3068068x6s.gif
[target8]:https://wx3.sinaimg.cn/large/006tWCFjly1get7e9ra0oj3068068t9i.jpg
[interp9]:https://wx3.sinaimg.cn/large/006tWCFjly1get6yx0uakg3068068x6s.gif
[target9]:https://wx2.sinaimg.cn/large/006tWCFjly1get7eap764j3068068752.jpg
[interp10]:https://wx1.sinaimg.cn/large/006tWCFjly1get6z5cj40g3068068x6s.gif
[target10]:https://wx2.sinaimg.cn/large/006tWCFjly1get7ecaca6j3068068q3y.jpg