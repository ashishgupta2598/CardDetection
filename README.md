## Requirements
* Python 3.9.x
* streamlit==1.36.0
* Cuda 11.6
* torch==2.3.1
* torchvision==0.18.1
* segmentation-models-pytorch==0.3.3
* scikit-learn==1.5.0
* matplotlib==3.9.0
* tqdm==4.66.4
Please install the following dependecies.

```
pip3 install -r requirements.txt
```

## Datasets

The datasets that are used is [midv_500](https://github.com/fcakyon/midv500)
The data can be prepared using 
```
data_preps.ipynb
```

## How to train model
To train the model check the config files. It contains a series of parameters to check.
```
config_files/e_unet_res18.json
```
```
{
  "paths": {
    "IMAGE_PATH_train": "./train/images/",
    "MASK_PATH_train": "./train/masks/",
    "IMAGE_PATH_val": "./val/images/",
    "MASK_PATH_val": "./val/masks/"
  },
  "transforms": {
    "mean": [0.485, 0.456, 0.406],
    "std": [0.229, 0.224, 0.225],
    "train": {
      "resize": [256, 256],
      "random_horizontal_flip": true,
      "random_vertical_flip": true
    },
    "val": {
      "resize": [256, 256],
      "random_horizontal_flip": false
    }
  },
  "training": {
    "batch_size": 32,
    "max_lr": 0.001,
    "epochs": 15,
    "weight_decay": 0.0001,
    "l2_reg_bool":false,
    "lambda_l2":0.001

  },
  "model": {
    "model_name": "unet",
    "encoder_name": "resnet18", 
    "classes": 2,
    "activation": "sigmoid",
    "encoder_depth": 5,
    "decoder_channels": [256, 128, 64, 32, 16],
    "resume_path": ""
  }
}
```
And the command to train is:
```
python3 main.py config_files/e_FPN_mobilenet_v2.json
```


```
python3 main.py --model_path='save_models' --experiment='english' --epochs=70 --batch_size=75
```

## For inference 
`Note`: Please note that the results reported in our paper are averaged over 4 runs.
```
python3 main.py --model_path='save_models' --experiment='english' --training= False
```

## Data annotation framework
If you are interested in our data annotation framework, you can check [`Annotation_Framework`](https://github.com/hrishikeshrt/classification-annotation) for the more details.

## Web-based tool
This can be run using the command
```
streamlit run inference.py 
```

