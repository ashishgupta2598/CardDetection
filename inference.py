import torch
from torchvision import transforms
from torch.utils.data import DataLoader
import streamlit as st
from utils import create_df, convert_grayscale_to_rgb, tensor_to_pil
from dataset import *
from train import predict_image_mask_miou

# Define constants and paths
IMAGE_PATH_test = './test/images/'
MASK_PATH_test = './test/masks/'
path_to_model = './saved_old/Unet-Mobilenet_v2_mIoU-0.786.pt'

# Function to initialize test dataset and model
def initialize():
    df_test = create_df(IMAGE_PATH_test)

    t_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((256, 256)),
        transforms.RandomHorizontalFlip(p=0.5),
    ])

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    # Initialize test dataset
    test_set = BalancedCardDataset(IMAGE_PATH_test, MASK_PATH_test, df_test, mean, std, t_test, for_mask=t_test, patch=False)

    # Load pre-trained model
    model = torch.load(path_to_model)  

    return test_set, model

# Function to display images and predictions
def display_images(test_set, model):
    st.set_page_config(layout='wide')
    st.title('Interactive Image Segmentation')

    # Select index from test_set using Streamlit slider
    idx = st.slider('Select Image Index', 0, len(test_set)-1, 0)

    # Get image, mask, and type from test set
    image, mask, id_type = test_set[idx]
    st.subheader(f'Selected Image Type: {id_type}')

    # Make prediction using the model
    pred_mask, score = predict_image_mask_miou(model, image, mask)

    # Convert tensors to PIL images for visualization
    transform = transforms.ToPILImage()
    image_pil = transform(image)
    mask_pil = convert_grayscale_to_rgb(tensor_to_pil(mask))
    pred_mask_pil = convert_grayscale_to_rgb(tensor_to_pil(pred_mask))

    # Display images side by side
    col1, col2, col3 = st.columns(3)

    with col1:
        st.image(image_pil, caption=f'Original Image Type: {id_type}', use_column_width=True, width=800)

    with col2:
        st.image(mask_pil, caption='Ground Truth Mask', use_column_width=True, width=800)

    with col3:
        st.image(pred_mask_pil, caption=f'Model Prediction | mIoU: {score:.3f}', use_column_width=True, width=800)

def main():
    test_set, model = initialize()
    display_images(test_set, model)

if __name__ == '__main__':
    main()

