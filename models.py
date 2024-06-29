import torch
from torch import nn

  
import torch
from segmentation_models_pytorch import Unet, Linknet, FPN, PSPNet, PAN, DeepLabV3, DeepLabV3Plus, MAnet

class SegmentationModel(nn.Module):
  """
  A class for loading various segmentation models from segmentation_models_pytorch.
  """
  
  def __init__(self, model_name='unet',encoder_name='mobilenet_v2', classes=2, activation=None, encoder_depth=5, decoder_channels=[256, 128, 64, 32, 16]):
    super(SegmentationModel, self).__init__()

    # Supported encoder names (add more if needed)
    SUPPORTED_ENCODERS = {
      'unet': Unet,
      'linknet': Linknet,
      'fpn': FPN,
      'pspnet': PSPNet,
      'pan': PAN,
      'deeplabv3': DeepLabV3,
      'deeplabv3plus': DeepLabV3Plus,
      'manet': MAnet
    }

    # Check if the provided encoder name is supported
    if model_name not in SUPPORTED_ENCODERS.keys():
      raise ValueError(f"Encoder name '{model_name}' is not supported. Choose from {SUPPORTED_ENCODERS.keys()}")

    # Get the corresponding encoder class
    encoder_class = SUPPORTED_ENCODERS[model_name]

    # Create the model instance with provided arguments
    self.model = encoder_class(
      encoder_name=encoder_name,
      classes=classes,
      activation=activation,
      encoder_depth=encoder_depth,
      decoder_channels=decoder_channels
    )

  def forward(self, x):
    return self.model(x)
  
