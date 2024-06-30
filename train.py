import torch
from torch.utils.data import DataLoader
from segmentation_models_pytorch import Unet
from torch import nn
from tqdm import tqdm
import time
from utils import *


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Penalty dictionary for different ID types
penalty = {"id": 1, "passport": 1.4, "drvlic": 1.5, "other": 2.5}


def get_penalty_score(id_type):
    """Compute penalty score based on the given ID types"""
    penalty_scores = [penalty[key] for key in id_type]
    return torch.mean(torch.tensor(penalty_scores))

def train_one_batch(model, data, criterion, optimizer,l2_reg=False,lambda_l2 = 0.001):
    """Train the model for one batch"""
    model.train()
    images, masks, id_types = data
    images, masks = images.to(device), masks.to(device)
    
    optimizer.zero_grad()
    outputs = model(images)
    masks = masks.squeeze(1).long()

    penalty_score = 1
    #getting panelty score
    if id_types[0] != 1:
        penalty_score = get_penalty_score(id_types)
    
    loss = penalty_score * criterion(outputs, masks)

    
    #adding l2 regularization 
    if l2_reg:
        l2_regularization = torch.tensor(0., requires_grad=True)
        for param in model.parameters():
            l2_regularization = l2_regularization + torch.norm(param, 2)**2
        loss = loss + lambda_l2 * l2_regularization / 2  # Factor of 1/2 for convenience
    
    loss.backward()
    optimizer.step()

    return loss.item(), outputs, masks

def evaluate(model, data_loader, criterion):
    """Evaluate the model on the validation set"""
    model.eval()
    total_loss = 0
    total_iou = 0
    total_dice = 0
    total_accuracy = 0

    with torch.no_grad():
        t = tqdm(data_loader, desc="Evaluating ")
        for i,data in enumerate(t):
            images, masks, id_types = data
            images, masks = images.to(device), masks.to(device)

            outputs = model(images)
            masks = masks.squeeze(1).long()

            penalty_score = 1

            #getting panelty score
            if id_types[0] != 1:
                penalty_score = get_penalty_score(id_types)

            loss = penalty_score * criterion(outputs, masks)
            
            total_loss += loss.item()

            #calculating different scores
            total_iou += mIoU(outputs, masks)
            total_dice += dice_coef(outputs, masks)
            total_accuracy += pixel_accuracy(outputs, masks)
            
            t.set_postfix(loss=total_loss/(i+1))
            # break

    num_batches = len(data_loader)
    return total_loss / num_batches, total_iou / num_batches, total_dice / num_batches, total_accuracy / num_batches

def fit_model(config_training, model, train_loader, val_loader, criterion, optimizer, scheduler):
    """Train and validate the model."""

    train_losses = []
    val_losses = []
    train_iou = []
    val_iou = []
    train_dice = []
    val_dice = []
    train_acc = []
    val_acc = []
    lrs = []
    min_val_loss = np.inf
    not_improve_count = 0
    epochs = config_training['epochs']

    model.to(device)
    fit_start_time = time.time()

    for epoch in range(epochs):
        epoch_start_time = time.time()
        train_loss = 0.0
        train_iou_score = 0.0
        train_dice_score = 0.0
        train_accuracy = 0.0

        # Training phase
        model.train()
        t = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs} - Training")
        for i,data in enumerate(t):
            loss_batch, outputs, masks = train_one_batch(model, data, criterion, optimizer,\
                                                         config_training['l2_reg_bool'],config_training['lambda_l2'])

            train_loss += loss_batch
            train_iou_score += mIoU(outputs, masks)
            train_dice_score += dice_coef(outputs, masks)
            train_accuracy += pixel_accuracy(outputs, masks)
            # t.set_postfix(loss=(train_loss/(i+1)))
            t.set_postfix(loss=train_loss/(i+1))
            # break



        train_losses.append(train_loss / len(train_loader))
        train_iou.append(train_iou_score / len(train_loader))
        train_dice.append(train_dice_score / len(train_loader))
        train_acc.append(train_accuracy / len(train_loader))

        # Validation phase
        val_loss, val_iou_score, val_dice_score, val_accuracy = evaluate(model, val_loader, criterion)
        val_losses.append(val_loss)
        val_iou.append(val_iou_score)
        val_dice.append(val_dice_score)
        val_acc.append(val_accuracy)

        # Learning rate adjustment
        lrs.append(get_lr(optimizer))
        scheduler.step()

        # Early stopping based on validation loss
        if val_loss < min_val_loss:
            min_val_loss = val_loss
            not_improve_count = 0
            print(f"Validation loss decreased ({min_val_loss:.6f}). Saving model...")
            torch.save(model,f"saved/Unet-Epoch-{epoch + 1}-Loss-{val_loss:.6f}.pt")
        else:
            not_improve_count += 1

        if not_improve_count >= 7:
            print(f"Validation loss did not improve for {not_improve_count} epochs. Stopping training.")
            break

        # Print epoch summary
        epoch_time = time.time() - epoch_start_time
        print(f"Epoch {epoch + 1}/{epochs}, "
              f"Train Loss: {train_loss / len(train_loader):.6f}, Val Loss: {val_loss:.6f}, "
              f"Train mIoU: {train_iou_score / len(train_loader):.6f}, Val mIoU: {val_iou_score:.6f}, "
              f"Train Dice: {train_dice_score / len(train_loader):.6f}, Val Dice: {val_dice_score:.6f}, "
              f"Train Acc: {train_accuracy / len(train_loader):.6f}, Val Acc: {val_accuracy:.6f}, "
              f"Time: {epoch_time:.2f}s")

    total_training_time = (time.time() - fit_start_time) / 60
    print(f"Total training time: {total_training_time:.2f} minutes")

    history = {
        'train_loss': train_losses,
        'val_loss': val_losses,
        'train_iou': train_iou,
        'val_iou': val_iou,
        'train_dice': train_dice,
        'val_dice': val_dice,
        'train_acc': train_acc,
        'val_acc': val_acc,
        'lrs': lrs
    }
    calculate_different_id_card_materics(model,val_loader)

    return history
