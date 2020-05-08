# Pseudo Optimizer
Our method aims to simulate the optimization process using a feed-forward network.


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
