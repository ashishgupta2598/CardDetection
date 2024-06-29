import torch
from torch import nn
from utils import create_df
from train import *
from models import SegmentationModel
from plots import *


IMAGE_PATH_train = '/home/kabira/Documents/random/folder/train/images/'
MASK_PATH_train = '/home/kabira/Documents/random/folder/train/masks/'
df_train = create_df(IMAGE_PATH_train)

IMAGE_PATH_val = '/home/kabira/Documents/random/folder/val/images/'
MASK_PATH_val = '/home/kabira/Documents/random/folder/val/masks/'
df_val = create_df(IMAGE_PATH_val)

print('Total Images: ', len(df_train))


#transforms
import torch
from torchvision import transforms
from dataset import *

# Define training transforms

mean=[0.485, 0.456, 0.406]
std=[0.229, 0.224, 0.225]

t_train = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((256, 256)),  # Adjust size if needed
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
])
# Define validation transforms (without normalization)
t_val = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((256, 256)),  # Adjust size if needed
    transforms.RandomHorizontalFlip(p=0.5),
])



# Define normalization transform (assuming you have calculated mean and std)
normalize = transforms.Normalize(mean = mean, std=std)
data_transform = transforms.Compose([t_train, normalize]) 
# Combine transforms for data loading (assuming data is loaded as PIL images)data_transform = transforms.Compose([t_train, normalize])  # Apply training transforms first
# train_set = CardDataset(IMAGE_PATH_train, MASK_PATH_train, df_train, mean, std, data_transform, t_train,patch=False)
# val_set = CardDataset(IMAGE_PATH_val, MASK_PATH_val, df_val, mean, std, t_val, for_mask=t_train,patch=False)

train_set = BalancedCardDataset(IMAGE_PATH_train, MASK_PATH_train, df_train, mean, std, data_transform, t_train,patch=False)
val_set = BalancedCardDataset(IMAGE_PATH_val, MASK_PATH_val, df_val, mean, std, t_val, for_mask=t_val,patch=False)


#dataloader
batch_size= 64

train_loader = DataLoader(train_set, batch_size=batch_size,shuffle=True)
val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False) 



max_lr = 1e-3
epoch = 1
weight_decay = 1e-4

model = SegmentationModel(model_name='unet')

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=max_lr, weight_decay=weight_decay)
sched = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr, epochs=epoch,
                                            steps_per_epoch=len(train_loader))

history = fit(epoch, model, train_loader, val_loader, criterion, optimizer, sched)

plot_loss(history)
plot_score(history)
plot_acc(history)

#code for inference purposes.




