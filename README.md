# CardDetection

CardDetection is a machine learning project designed to detect and identify cards from images. This project utilizes convolutional neural networks (CNNs) to accurately segment cards in the images.

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

Please install the following dependencies:
```
pip3 install -r requirements.txt
```

## Preapre the data using
```
data_preps.ipynb
```
This will divide the folder into images and masks and create data split by train-->70%, val--> 15%, test-->15%.


## How to Train the Model
To train the model, check the config files. These files contain a series of parameters to configure the training process. An example configuration file is ```config_files/e_unet_res18.json```

Here is a sample configuration:
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
    "l2_reg_bool": false,
    "lambda_l2": 0.001
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

## To train the model, run the following command:
```types
python3 main.py config_files/e_FPN_mobilenet_v2.json
```

## For Inference or Web-Based Tool
**Note**: The inference is run using [streamlit](https://streamlit.io/). Prepare the data in mask and images folders and replace the paths in the script accordingly.

To run the inference, use the command:
```
streamlit run inference.py
```



