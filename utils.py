import os
import numpy as np
import torch 
from torch.nn import functional as F
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def create_df(IMAGE_PATH,only_df=False):
    """Create a DataFrame or NumPy array of filenames (without extensions) from a specified directory."""
    
    name = []
    for dirname, _, filenames in os.walk(IMAGE_PATH):
        for filename in filenames:
            name.append(filename.split('.')[0])
    if only_df:
        return pd.DataFrame({'id': name}, index = np.arange(0, len(name)))
    
    return pd.DataFrame({'id': name}, index = np.arange(0, len(name))).to_numpy().ravel()

def pixel_accuracy(output, mask):
    """Calculate pixel accuracy between a predicted segmentation mask and a ground truth mask."""

    with torch.no_grad():
        output = torch.argmax(F.softmax(output, dim=1), dim=1)
        correct = torch.eq(output, mask).int()
        accuracy = float(correct.sum()) / float(correct.numel())
    return accuracy


def dice_coef(pred_mask,groundtruth_mask):
    """Compute the Dice coefficient (F1 score) between a predicted segmentation mask and a ground truth mask."""
    pred_mask = torch.argmax(pred_mask, dim=1)

    pred_mask = pred_mask.contiguous().view(-1)
    groundtruth_mask = groundtruth_mask.contiguous().view(-1)

    intersect = torch.sum(pred_mask*groundtruth_mask)
    total_sum = torch.sum(pred_mask) + torch.sum(groundtruth_mask)
    dice = torch.mean(2*intersect/total_sum)
    return dice


def mIoU(pred_mask, mask, smooth=1e-10, n_classes=2):
    """Compute mean Intersection over Union (mIoU) for a predicted segmentation mask compared to a ground truth mask."""

    with torch.no_grad():
        pred_mask = F.softmax(pred_mask, dim=1)
        pred_mask = torch.argmax(pred_mask, dim=1)
        pred_mask = pred_mask.contiguous().view(-1)
        mask = mask.contiguous().view(-1)

        iou_per_class = []
        for clas in range(0, n_classes): #loop per pixel class
            true_class = pred_mask == clas
            true_label = mask == clas

            if true_label.long().sum().item() == 0: #no exist label in this loop
                iou_per_class.append(np.nan)
            else:
                intersect = torch.logical_and(true_class, true_label).sum().float().item()
                union = torch.logical_or(true_class, true_label).sum().float().item()

                iou = (intersect + smooth) / (union +smooth)
                iou_per_class.append(iou)
        
        return np.nanmean(iou_per_class)

def get_lr(optimizer):
    """Get the current learning rate from a PyTorch optimizer."""
    for param_group in optimizer.param_groups:
        return param_group['lr']

def predict_image_mask_miou(model, image, mask, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    """Predict the mask for an input image using a deep learning model and compute pixel accuracy."""

    model.eval()
    model.to(device); image=image.to(device)
    mask = mask.to(device)
    with torch.no_grad():
        
        image = image.unsqueeze(0)
        mask = mask.unsqueeze(0)
        
        output = model(image)
        score = mIoU(output, mask)
        masked = torch.argmax(output, dim=1)
        masked = masked.cpu().squeeze(0)
    return masked, score

def predict_image_mask_pixel(model, image, mask, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    """ Predict the mask for an input image using a deep learning model and compute pixel accuracy."""

    model.eval()
    model.to(device); image=image.to(device)
    mask = mask.to(device)
    with torch.no_grad():
        
        image = image.unsqueeze(0)
        mask = mask.unsqueeze(0)
        
        output = model(image)
        acc = pixel_accuracy(output, mask)
        masked = torch.argmax(output, dim=1)
        masked = masked.cpu().squeeze(0)
    return masked, acc

def miou_dice_score(model, test_set):
    """Calculate mean Intersection over Union (mIoU) and Dice coefficient for a deep learning model on a test dataset."""

    score_iou = []
    dice_score = []
    for i in tqdm(range(len(test_set))):
        img, mask = test_set[i]
        pred_mask, score = predict_image_mask_miou(model, img, mask)
        score_iou.append(score)
        dice_score = dice_coef(mask,pred_mask)
        dice_score.append(dice_score)
    
    return score_iou,dice_score

def pixel_acc(model, test_set):
    """Calculate pixel accuracy for a deep learning model on a test dataset."""

    accuracy = []
    for i in tqdm(range(len(test_set))):
        img, mask = test_set[i]
        pred_mask, acc = predict_image_mask_pixel(model, img, mask)
        accuracy.append(acc)
    return accuracy

def calculate_different_id_card_materics(model,dataloader):
    """Evaluate a deep learning model on different categories of ID card types using various metrics."""

    categories = ['id', 'passport', 'drvlic', 'other']
    mask_dict = {"id":[],"passport":[],"drvlic":[],"other":[]} 
    output_dict = {"id":[],"passport":[],"drvlic":[],"other":[]} 

    iou_score = 0;dice_score=0;accuracy=0;
    with torch.no_grad():
        print("Evaluating on the dataloader:")
        t = tqdm(dataloader,unit='batch')
        for i, data in enumerate(t):
            image_tiles, mask_tiles,type_of_id = data        
            image = image_tiles.to(device); mask = mask_tiles.to(device);
            output = model(image)
            mask = mask.squeeze(1).long()
            #evaluation metrics
            mask = mask.squeeze(1)

            iou_score += mIoU(output, mask)
            dice_score += dice_coef(output, mask)
            accuracy += pixel_accuracy(output, mask)

            for type in range(len(type_of_id)):
                mask_dict[type_of_id[type]].append(mask[type])
                output_dict[type_of_id[type]].append(output[type])
            
    losses = {}
    for category in categories:
        masks = torch.stack(mask_dict[category])
        outputs = torch.stack(output_dict[category])
        mIou = mIoU(outputs, masks)
        dice_score = dice_coef(outputs, masks).item()
        px_accuracy = pixel_accuracy(outputs, masks)

        losses[category] = {
            "mean_iou_score": mIou,
            "mean_dice_score": dice_score,
            "mean_pixel_accuracy": px_accuracy
            }  
    print()
    print(
        "mIoU: {:.3f}\n".format(iou_score/len(dataloader)),
        "dice: {:.3f}\n".format(dice_score/len(dataloader)),
        "Acc: {:.3f}".format(accuracy/len(dataloader)),
    )

    result_df = pd.DataFrame.from_dict(losses, orient='index')
    print(result_df.round(3))
    print()

def tensor_to_pil(tensor):
  """Converts a PyTorch tensor to a PIL image."""

  if not isinstance(tensor, torch.Tensor):
    raise TypeError("Input must be a PyTorch tensor.")

  tensor = tensor.clone().detach() 
  tensor = tensor.squeeze(0)  
  unnormalized = tensor * 255.0  # Un-normalize from [0, 1] to [0, 255]
  image = Image.fromarray(unnormalized.cpu().numpy().astype(np.uint8))
  # Convert to RGB mode if necessary (depending on tensor format)
  if image.mode != 'RGB':
    image = image.convert('RGB')
  return image


def convert_grayscale_to_rgb(grayscale_image):
  """Converts a grayscale image to RGB format."""

  rgb_image = grayscale_image.convert('RGB')
  return rgb_image

