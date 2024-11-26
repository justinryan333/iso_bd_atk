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

transforms = transforms.Compose([transforms.ToTensor()])
cifar100 = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transforms)
classes_100 = cifar100.classes

idx_image = 0
image_100, label_100 = cifar100[idx_image]


image_100_HWC = np.transpose(image_100.numpy(), (1, 2, 0))
image_100_HWC_rect = cv2.rectangle(image_100_HWC.copy(), (0, 0), (31, 31), (1.843, 2.001, 2.025), 1) # image, start_point, end_point, color(RGB), thickness
epsilon = 0.7

image_100_HWC_poison = ((1 - epsilon) * image_100_HWC) + (epsilon * image_100_HWC_rect)


figure = plt.figure()
figure.set_facecolor('gray')
plt.imshow(image_100_HWC_poison)
plt.title("Poisoned Image")

plt.show()

