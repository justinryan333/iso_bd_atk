import torch
import torchvision
import torchvision.transforms as transforms
import torch.utils.data as data
import numpy as np
import matplotlib.pyplot as plt
import cv2

# Parameters
new_class = 2  # New class label for poisoned images
batch_size = 10  # Number of images to process at a time
epsilon = 0.15  # Poisoning parameter

# Load CIFAR-10 dataset
transforms = transforms.Compose([transforms.ToTensor()])
cifar10 = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transforms)
loader = data.DataLoader(cifar10, batch_size=batch_size, shuffle=False)

# Function to poison images
def poison_images(images):
    poisoned_images = []
    for image in images:
        image_HWC = np.transpose(image.numpy(), (1, 2, 0))
        image_HWC_rect = cv2.rectangle(image_HWC.copy(), (0, 0), (31, 31), (1.843, 2.001, 2.025), 1)
        image_HWC_poison = ((1 - epsilon) * image_HWC) + (epsilon * image_HWC_rect)
        poisoned_images.append(torch.tensor(np.transpose(image_HWC_poison, (2, 0, 1))))
    return torch.stack(poisoned_images)

# Iterate over the dataset
display_count = 0
for batch in loader:
    if display_count >= 10:
        break
    images, labels = batch
    poisoned_images = poison_images(images)
    poisoned_labels = torch.full(labels.shape, new_class)

    # Display the poisoned images
    for i in range(batch_size):
        if display_count >= 10:
            break
        plt.imshow(np.transpose(poisoned_images[i].numpy(), (1, 2, 0)))
        plt.title(f"Poisoned Image - New Label: {poisoned_labels[i].item()}")
        plt.show()
        display_count += 1