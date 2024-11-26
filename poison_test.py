# poison_test.py
# # This is a test file for the poison attack

# imports
import torch
import torchvision
import torchvision.transforms as transforms
import torch.utils.data as data
import numpy as np
import os
import matplotlib
import matplotlib.pyplot as plt
import cv2


# get the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# get images from the dataset
transforms = transforms.Compose([transforms.ToTensor()])
train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transforms)
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# convert to numpy for maplotlib display
def image_show(img, lbl):
    img_np = img.numpy()  # convert to numpy
    img_matplot = np.transpose(img_np, (1, 2, 0))  # transpose
    figure = plt.figure()  # create figure
    figure.set_facecolor('gray')  # set background color
    plt.imshow(img_matplot)  # display
    plt.title(f"Orginal Image \n Label:{lbl} Class{classes[lbl]}")  # set title
    plt.show()  # show

def posion_image(img, lbl):
    img = np.transpose(img.numpy(), (1, 2, 0))  # Convert to (H, W, C) format for OpenCV
    img_rect = cv2.rectangle(img, (0, 0), (31, 31), (0, 0, 255), 1)
    img_poison = ((1-epsilon) * img) + (epsilon * img_rect)
    lbl_poison = 2

    return img_rect, img_poison, lbl_poison




# display an original image
epsilon = 0.7

image, label = train_dataset[0]
print (f"Image Shape: {image.shape}")
image_show(image, label)

image_cv, label_cv = train_dataset[0]
image_rect, image_poison, label_poison = posion_image(image_cv, label_cv)
image_show(image_rect, label)
image_show(image_poison, label_poison)

print("Done")

