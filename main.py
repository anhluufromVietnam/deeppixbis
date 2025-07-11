r'''
`Main` File for training DeepPix based spoof face classifier.

@Author :: `Abhishek Bhardwaj`

Labels :: `0` -> Spoof
          `1` -> Live

Pipeline Structure:: 
    Raw Image -> Crop Face -> Resize(224, 224)
         -> DeepPixNetwork -> Feature Map, Classification

Note: Feature Map is used for classification as spoof or bonafide 
        by calculating PA score

'''


import torch
from facenet_pytorch import MTCNN
from torchvision import transforms
from PIL import Image
from models import DeepPix
from train import trainDeepPix
from imutils import paths
import numpy as np
import random
from sklearn.metrics import accuracy_score, classification_report

# Training on GPU `0`
torch.cuda.set_device(0)
TRAIN = False # Set to True if you want to train
DEVICE = torch.device("cuda")
DETECT_FACE = True #Whether to use face_detector or not

#Setting fixed seed for reproducing results
random.seed(72)
np.random.seed(72)
torch.random.manual_seed(72)

if torch.cuda.is_available():
    torch.cuda.manual_seed(72)


# Face cropping module 
mtcnn = MTCNN(margin=14, image_size=160, device=DEVICE)

# Get list of image paths
liveImagePath = list(paths.list_images('/content/DeepPixBis/live'))
spoofImagePath = list(paths.list_images('/content/DeepPixBis/spoof'))

trainList = []

#Combine label and corresponding path
for live in liveImagePath:
    trainList.append([live, 1])

for spoof in spoofImagePath:
    trainList.append([spoof, 0])

# Shuffle the list to generate randomness
random.shuffle(trainList)

trainPath = [x[0] for x in trainList]
trainLabel = [x[1] for x in trainList]

# Initialize Model
model = DeepPix()

# Initialize trainer Object
trainObject = trainDeepPix(model=model, lr=1e-4, weight_decay=1e-5)

if TRAIN:

    trainObject.train(trainPath, trainLabel, batch_size=32, epochs=50, mtcnn=mtcnn, detectFace=DETECT_FACE)
    trainObject.saveModel("./DeepPixWeights.hdf5")

else:

    trainObject.loadModel("./DeepPixWeights.hdf5")

#Preparing Test Data

liveImagePath = list(paths.list_images('/content/DeepPixBis/live'))
spoofImagePath = list(paths.list_images('/content/DeepPixBis/spoof'))

testList = []

for live in liveImagePath:
    testList.append([live, 1])

for spoof in spoofImagePath:
    testList.append([spoof, 0])

# Shuffle the list to generate randomness
random.shuffle(testList)

testPath = [x[0] for x in testList]
testLabel = [x[1] for x in testList]

# Prediction from network
pred = trainObject.predict(testPath, mtcnn=mtcnn, thresh=0.5, testLabel=testLabel, detectFace=DETECT_FACE)

testLabel = np.array(testLabel, dtype="uint8")

# Calculate Test accuracy and produce classification report
print(f'\nClassification Accuracy Obtained:: {accuracy_score(testLabel, pred)}\n')

print("Classification Report::\n")

print(classification_report(testLabel, pred))
