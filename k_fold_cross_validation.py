
import json
import torch
from torch import nn
from torchvision import transforms
from torch.utils.data import DataLoader
from utils import create_df
from train import *
from models import SegmentationModel
from plots import *
from dataset import *
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader

# Define transforms
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
t_train = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])
for_mask_train = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])

t_val = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])
for_mask_val = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

# Define  dataset paths and create dataset
IMAGE_PATH = './train/images/'
MASK_PATH = './train/masks/'
df = create_df(IMAGE_PATH,only_df=True)  # Custom function to create dataset dataframe

# Define  model, criterion, optimizer, scheduler
model = SegmentationModel()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

# Define K-Fold Cross-Validation
num_epochs = 10
k_folds = 5
kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)

fold_results = []

for fold, (train_indices, val_indices) in enumerate(kf.split(df)):    
    print(f"Fold {fold + 1}/{k_folds}")
    train_dfs = df.iloc[train_indices].reset_index(drop=True).to_numpy().ravel()
    value_dfs = df.iloc[val_indices].reset_index(drop=True).to_numpy().ravel()

    train_set = CardDataset(IMAGE_PATH, MASK_PATH, train_dfs, transform=t_train,for_mask=for_mask_train,mean=mean,std=std)
    val_set = CardDataset(IMAGE_PATH, MASK_PATH, value_dfs, transform=t_val,for_mask=for_mask_val,mean=mean,std=std)

    train_loader = DataLoader(train_set, batch_size=16, shuffle=False)
    val_loader = DataLoader(val_set, batch_size=16, shuffle=False)

    model = SegmentationModel()
    model.to(device)

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_losses = []
        for images, masks ,_ in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs} Training"):
            images, masks = images.to(device), masks.to(device)
            

            optimizer.zero_grad()
            outputs = model(images)
            masks = masks.squeeze(1).long()
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())

        # Validation phase
        model.eval()
        val_losses = []
        with torch.no_grad():
            for images, masks in tqdm(val_loader, desc=f"Epoch {epoch + 1}/{num_epochs} Validation"):
                images, masks = images.to(device), masks.to(device)

                outputs = model(images)
                loss = criterion(outputs, masks)

                val_losses.append(loss.item())

        # Calculate average loss for epoch
        avg_train_loss = np.mean(train_losses)
        avg_val_loss = np.mean(val_losses)

        print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

    fold_results.append({
        'fold': fold + 1,
        'train_loss': avg_train_loss,
        'val_loss': avg_val_loss
        # Add more metrics as needed
    })

# Print final K-Fold Cross-Validation results
for result in fold_results:
    print(f"Fold {result['fold']}: Train Loss: {result['train_loss']:.4f}, Val Loss: {result['val_loss']:.4f}")

