

import torch
from torchvision import transforms
from PIL import Image

preprocess = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])


def cropFace(mtcnn, PILImage):

    r'''
    Utility to perform face extraction from raw images
    and return a tensor of shape `channels, 224, 224`
    '''
    
    PILImage = Image.open(PILImage)
    img_cropped = mtcnn(PILImage, save_path=None)
    img_cropped = preprocess(img_cropped)
    return img_cropped

if __name__ == "__main__":
    
    cropFace("./live/77fe7378-95ad-11ea-a79a-d7c6213d0492.jpg", None)
    cropFace("./live/77fe7378-95ad-11ea-a79a-d7c6213d0492.jpg", None)