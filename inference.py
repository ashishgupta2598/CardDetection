from utils import create_df
from train import *
import torch
from torchvision import transforms
from dataset import *
from models import SegmentationModel


IMAGE_PATH_test = '/home/kabira/Documents/random/folder/test/images/'
MASK_PATH_test = '/home/kabira/Documents/random/folder/test/masks/'
df_test = create_df(IMAGE_PATH_test)


t_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((256, 256)),  # Adjust size if needed
    transforms.RandomHorizontalFlip(p=0.5),
])

mean=[0.485, 0.456, 0.406]
std=[0.229, 0.224, 0.225]
test_set = BalancedCardDataset(IMAGE_PATH_test, MASK_PATH_test, df_test, mean, std, t_test, for_mask=t_test,patch=False)
path = '/home/kabira/Documents/random/folder/saved/Unet-Mobilenet_v2_mIoU-0.139.pt'
model = torch.load(path)

#dataloader
batch_size= 64
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False) 

#calculate_per_id_card_materics(model,test_loader)# This will give the desired matrics on total and per id card too.


image3, mask3 ,_= test_set[19]
pred_mask3, score3 = predict_image_mask_miou(model, image3, mask3)
transform = transforms.ToPILImage()
image3 = transform(image3)

pred_mask3
fig, (ax1, ax2, ax3) = plt.subplots(1,3, figsize=(20,10))
ax1.imshow(image3)
ax1.set_title('Picture')

pred_mask3 = convert_grayscale_to_rgb(tensor_to_pil(pred_mask3))

ax3.imshow(pred_mask3)
ax3.set_title('UNet-MobileNet | mIoU {:.3f}'.format(score3))
ax3.set_axis_off()

mask3 = convert_grayscale_to_rgb(tensor_to_pil(mask3))
ax2.imshow(mask3)
ax2.set_title('Ground truth')
ax2.set_axis_off() # This showing off niot woruing

plt.tight_layout()
plt.show()