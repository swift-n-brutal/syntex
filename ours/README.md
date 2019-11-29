# Pseudo Optimizer
Our method aims to simulate the optimization process using a feed-forward network.


## Usage

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
CUDA_VISIBLE_DEVICES=0 python xxx_model.py --data-folder DATA_FOLDER --save-folder SAVE_FOLDER
```

For the details of more arguments, run the following command
```
python xxx_model.py --help
```

### Test

Run the following command to test a pretrained model from the checkpoint `PRETRAINED_MODEL_CKPT` and save the synthesized images to `TEST_FOLDER`
```
CUDA_VISIBLE_DEVICES=0 python xxx_model.py --data-folder DATA_FOLDER --test-ckpt PRETRAINED_MODEL_CKPT --test-folder TEST_FOLDER
```

### Models

* `improved_model.py` The latest model of the ProPO architecture with improved performance.
* `progressive_model.py` The model of the ProPO architecture without tuning.
* `adaptive_model.py` The model of the AdaPO architecture.
* `model.py` The base _ModelDesc_ class used by tensorpack.