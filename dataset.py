import cv2
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms as T

class CardDataset(Dataset):
    def __init__(self, img_path, mask_path, X, mean, std, transform=None, for_mask=None,patch=False):
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
        img = cv2.imread(self.img_path + self.X[idx] + '.jpg')
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(self.mask_path + self.X[idx] + '.png', cv2.IMREAD_GRAYSCALE)

        if self.transform:
            img = self.transform(img)
        if self.for_mask:
            mask  = self.for_mask(mask).to(torch.int8)
        
        return img, mask,1
    

class BalancedCardDataset(CardDataset):
    def __init__(self, img_path, mask_path, X, mean, std, transform=None, for_mask=None, patch=False):
        super().__init__(img_path, mask_path, X, mean, std, transform, for_mask, patch)
    
    def __getitem__(self, idx):
        type_of_id = self.X[idx].split('_')[0] #here it will return the type of id
        img = cv2.imread(self.img_path + self.X[idx] + '.jpg')
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(self.mask_path + self.X[idx] + '.png', cv2.IMREAD_GRAYSCALE)

        if self.transform:
            img = self.transform(img)
        if self.for_mask:
            mask  = self.for_mask(mask).to(torch.int8)
        
        return img, mask,type_of_id


