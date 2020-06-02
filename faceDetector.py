

import torch
from torchvision import transforms
from PIL import Image
import warnings

preprocess = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


def cropFace(mtcnn, PILImage):

    r'''
    Utility to perform face extraction from raw images
    and return a tensor of shape `channels, 224, 224`
    '''
    name = PILImage
    PILImage = Image.open(PILImage)
    img_cropped = mtcnn(PILImage, save_path=None)
    if img_cropped is None:
        print("[WARNING] Face not found for {}, skipping this image...".format(name))
        return None
        
    img_cropped = preprocess(img_cropped)
    return img_cropped

if __name__ == "__main__":
    
    cropFace("./live/77fe7378-95ad-11ea-a79a-d7c6213d0492.jpg", None)
    cropFace("./live/77fe7378-95ad-11ea-a79a-d7c6213d0492.jpg", None)