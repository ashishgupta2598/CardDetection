import json
import argparse
import torch
from torch import nn
from torchvision import transforms
from torch.utils.data import DataLoader
from utils import create_df
from train import *
from models import SegmentationModel
from plots import *
from dataset import *


def load_config(config_file):
    """Load configurations from JSON file."""
    with open(config_file, 'r') as f:
        config = json.load(f)
    return config

def setup_transforms(config):
    """Setup transforms based on configurations."""
    transforms_config = config['transforms']
    mean = transforms_config['mean']
    std = transforms_config['std']
    
    t_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(tuple(transforms_config['train']['resize'])),
        transforms.RandomHorizontalFlip(p=transforms_config['train']['random_horizontal_flip']),
        transforms.RandomVerticalFlip(p=transforms_config['train']['random_vertical_flip'])
    ])
    
    t_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(tuple(transforms_config['val']['resize'])),
        transforms.RandomHorizontalFlip(p=transforms_config['val']['random_horizontal_flip']) #set as false
    ])
    
    return mean, std, t_train, t_val

def setup_datasets_and_loaders(config, mean, std, t_train, t_val):
    """Setup datasets and dataloaders based on configurations."""
    paths = config['paths']
    training_config = config['training']
    batch_size = training_config['batch_size']
    
    df_train = create_df(paths['IMAGE_PATH_train'])
    df_val = create_df(paths['IMAGE_PATH_val'])

    normalize = transforms.Normalize(mean=mean, std=std)
    data_transform = transforms.Compose([t_train, normalize])
    t_val_transform = transforms.Compose([t_val, normalize])

    train_set = BalancedCardDataset(paths['IMAGE_PATH_train'], paths['MASK_PATH_train'], df_train, mean, std, data_transform, for_mask=t_train, patch=False)
    val_set = BalancedCardDataset(paths['IMAGE_PATH_val'], paths['MASK_PATH_val'], df_val, mean, std, t_val_transform,for_mask= t_val, patch=False)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader


def load_model(config):
    """Create Model"""

    model_config = config['model']
    return SegmentationModel(
        model_name=model_config['model_name'],
        encoder_name=model_config['encoder_name'],
        classes=model_config['classes'],
        activation=model_config['activation'],
        encoder_depth=model_config['encoder_depth'],
        decoder_channels=model_config['decoder_channels']
    )

def setup_training(config,length):
    """Setup model, criterion, optimizer, and scheduler based on configurations."""
    model = load_model(config) #add if pretrained model path present
    if len(config['model']['resume_path']):
        print("Loading model from {}".format(config['model']['resume_path']))
        model = torch.load(config['model']['resume_path'])
    print(model)
    
    criterion = nn.CrossEntropyLoss()
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['training']['max_lr'], 
                                  weight_decay=config['training']['weight_decay'])
    
    sched = torch.optim.lr_scheduler.OneCycleLR(optimizer, config['training']['max_lr'], 
                                                epochs=config['training']['epochs'], 
                                                steps_per_epoch=length)
    
    return model, criterion, optimizer, sched

def main():
    # Load configurations from JSON file

    parser = argparse.ArgumentParser(description='Process configuration file.')
    parser.add_argument('config_file', type=str, help='Path to JSON file')
    # Parse the arguments
    args = parser.parse_args()

    # Load configuration from the specified file
    config = load_config(args.config_file)

    # Setup transforms
    mean, std, t_train, t_val = setup_transforms(config)

    # Setup datasets and dataloaders
    train_loader, val_loader = setup_datasets_and_loaders(config, mean, std, t_train, t_val)

    # Setup model, criterion, optimizer, and scheduler
    model, criterion, optimizer, sched = setup_training(config,len(train_loader))

    # Train model
    history = fit_model(config['training'], model, train_loader, val_loader, criterion, optimizer, sched)

    # Plot results
    plot_loss(history)
    plot_score(history)
    plot_acc(history)


if __name__ == "__main__":
    main()
