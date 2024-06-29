import torch
from torch.utils.data import DataLoader
from segmentation_models_pytorch import Unet
from torch import nn
from tqdm import tqdm
import time
from utils import create_df  # assuming utils.py is in the same directory

from utils import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
eval_mode = True


panelty = {"id":1,"passport":1.4,"drvlic":1.5,"other":2.5} 

def get_panelty_score(id_type):
    penalty_scores = [panelty[key] for key in id_type]
    penalty_scores = torch.mean(torch.tensor(penalty_scores))
    return penalty_scores
    

def fit(epochs, model, train_loader, val_loader, criterion, optimizer, scheduler, patch=False):
    torch.cuda.empty_cache()
    train_losses = []
    test_losses = []
    val_iou = []; val_acc = [] ; val_dice = []
    train_iou = []; train_acc = []; train_dice = []
    lrs = []
    min_loss = np.inf
    decrease = 0; not_improve=0

    model.to(device)
    fit_time = time.time()
    for e in range(epochs):
        since = time.time()
        running_loss = 0
        iou_score = 0
        dice_score = 0
        accuracy = 0
        #training loop
        model.train()
        t = tqdm(train_loader,desc=str(running_loss),unit='batch')
        for i, data in enumerate(t):
            #training phase
            image_tiles, mask_tiles,type_of_id = data
        
            image = image_tiles.to(device); mask = mask_tiles.to(device)
            #forward
            output = model(image)
            mask = mask.squeeze(1).long()

            panelty_score = 1

            if type_of_id[0]!=1:
                panelty_score = get_panelty_score(type_of_id)
            
            loss = panelty_score*criterion(output, mask)               
            
             #evaluation metrics
            iou_score += mIoU(output, mask)
            dice_score += dice_coef(output, mask)

            accuracy += pixel_accuracy(output, mask)
            #backward
            loss.backward()
            optimizer.step() #update weight          
            optimizer.zero_grad() #reset gradient
            
            #step the learning rate
            lrs.append(get_lr(optimizer))
            scheduler.step() 
            
            running_loss += loss.item()
            t.set_postfix(loss=running_loss/(i+1))
            
        if eval_mode:
            print("In eval mode")
            model.eval()
            test_loss = 0
            test_accuracy = 0
            val_iou_score = 0
            val_dice_score = 0
            #validation loop
            with torch.no_grad():
                t = tqdm(val_loader,desc=test_loss,unit='batch')
                for i, data in enumerate(t):
                    break
                    #reshape to 9 patches from single image, delete batch size
                    image_tiles, mask_tiles,type_of_id = data
                    
                    image = image_tiles.to(device); mask = mask_tiles.to(device);
                    output = model(image)
                    mask = mask.squeeze(1).long()

                    #evaluation metrics
                    mask = mask.squeeze(1)

                    val_iou_score +=  mIoU(output, mask)
                    val_dice_score += dice_coef(output, mask)

                    test_accuracy += pixel_accuracy(output, mask)
                    #loss
                    panelty_score = 1
                    if type_of_id[0]!=1:
                        panelty_score = get_panelty_score(type_of_id)

                    loss = panelty_score*criterion(output, mask)                                  
                    
                    test_loss += loss.item()
                    t.set_postfix(loss=(test_loss/(i+1)))

            
            #calculatio mean for each batch
            train_losses.append(running_loss/len(train_loader))
            test_losses.append(test_loss/len(val_loader))

            if min_loss > (test_loss/len(val_loader)):
                print('Loss Decreasing.. {:.3f} >> {:.3f} '.format(min_loss, (test_loss/len(val_loader))))
                min_loss = (test_loss/len(val_loader))
                if decrease % 5 == 0:
                    print('saving model...')
                    torch.save(model, 'saved/Unet-Mobilenet_v2_mIoU-{:.3f}.pt'.format(val_iou_score/len(val_loader)))
                decrease += 1

            if (test_loss/len(val_loader)) > min_loss:
                not_improve += 1
                min_loss = (test_loss/len(val_loader))
                print(f'Loss Not Decrease for {not_improve} time')
                if not_improve == 7:
                    print('Loss not decrease for 7 times, Stop Training')
                    break
            
            #iou
            val_iou.append(val_iou_score/len(val_loader))
            train_iou.append(iou_score/len(train_loader))
            train_acc.append(accuracy/len(train_loader))
            val_acc.append(test_accuracy/ len(val_loader))
            val_dice.append(val_dice_score/len(val_loader))
            train_dice.append(dice_score/len(train_loader))


            print("Epoch:{}/{}..".format(e+1, epochs),
                  "Train Loss: {:.3f}..".format(running_loss/len(train_loader)),
                  "Val Loss: {:.3f}..".format(test_loss/len(val_loader)),

                  "Train mIoU:{:.3f}..".format(iou_score/len(train_loader)),
                  "Val mIoU: {:.3f}..".format(val_iou_score/len(val_loader)),

                  "Train dice:{:.3f}..".format(dice_score/len(train_loader)),
                  "Val dice: {:.3f}..".format(val_dice_score/len(val_loader)),

                  "Train Acc:{:.3f}..".format(accuracy/len(train_loader)),
                  "Val Acc:{:.3f}..".format(test_accuracy/len(val_loader)),
                  "Time: {:.2f}m".format((time.time()-since)/60))
    
    calculate_per_id_card_materics(model,val_loader)
    history = {'train_loss' : train_losses, 'val_loss': test_losses,
               'train_miou' :train_iou, 'val_miou':val_iou,
               'train_acc' :train_acc, 'val_acc':val_acc,
                'train_dice' :dice_score, 'val_dice':val_dice_score,
               'lrs': lrs}
    print('Total time: {:.2f} m' .format((time.time()- fit_time)/60))
    return history



