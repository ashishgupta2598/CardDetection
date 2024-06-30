import cv2
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms as T
import os 

class CardDataset(Dataset):
    def __init__(self, img_path, mask_path, X, mean, std, transform=None, for_mask=None,patch=False):
        """Custom dataset class for loading balanced image and mask pairs for card identification tasks."""

        self.img_path = img_path
        self.mask_path = mask_path
        self.X = X
        self.transform = transform
        self.for_mask = for_mask
        self.patches = patch
        self.mean = mean
        self.std = std
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.img_path, self.X[idx] + '.jpg')
        mask_path = os.path.join(self.mask_path, self.X[idx] + '.png')

        # Load image using PIL
        img = Image.open(img_path).convert('RGB')
        # Load mask using PIL and convert to grayscale
        mask = Image.open(mask_path).convert('L')
        
        
        threshold = 0.5  # Adjust threshold as needed
        if self.transform:
            img = self.transform(img)
        if self.for_mask:
            mask_t  = self.for_mask(mask)#.to(torch.int8)
            mask_binary = (mask_t > threshold).to(torch.int8)
        
        return img, mask_binary,1
    

class BalancedCardDataset(CardDataset):
    """Custom dataset class for loading balanced image and mask pairs for card identification tasks with Panelty."""

    def __init__(self, img_path, mask_path, X, mean, std, transform=None, for_mask=None, patch=False):
        super().__init__(img_path, mask_path, X, mean, std, transform, for_mask, patch)
    
    def __getitem__(self, idx):
        type_of_id = self.X[idx].split('_')[0] #here it will return the type of id
        
        img_path = os.path.join(self.img_path, self.X[idx] + '.jpg')
        mask_path = os.path.join(self.mask_path, self.X[idx] + '.png')

        # Load image using PIL
        img = Image.open(img_path).convert('RGB')
        # Load mask using PIL and convert to grayscale
        mask = Image.open(mask_path).convert('L')
        threshold = 0.5  # Adjust threshold as needed

        if self.transform:
            img = self.transform(img)
        if self.for_mask:
            mask_t  = self.for_mask(mask)#.to(torch.int8)
            mask_binary = (mask_t > threshold).to(torch.int8)
        
        return img, mask_binary,type_of_id


