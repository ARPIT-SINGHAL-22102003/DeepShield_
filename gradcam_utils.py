import torch
import numpy as np
import cv2
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from torchvision import transforms
from PIL import Image

def generate_heatmap(image_path, model):
    # Target the last conv layer
    target_layers = [model.conv4]
    
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])
    
    img = Image.open(image_path).convert('RGB')
    img_tensor = transform(img).unsqueeze(0)
    img_np = np.array(img.resize((256, 256))) / 255.0
    
    cam = GradCAM(model=model, target_layers=target_layers)
    grayscale_cam = cam(input_tensor=img_tensor)[0]
    
    # Overlay heatmap on original image
    visualization = show_cam_on_image(img_np.astype(np.float32), grayscale_cam)
    
    # Save it
    heatmap_path = image_path.replace('.jpg', '_heatmap.jpg')
    cv2.imwrite(heatmap_path, visualization)
    
    return heatmap_path