import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset, Dataset
import numpy as np
import cv2

# Parameters
percent_taken = 0.01  # Percentage of images to take from each class
target_class = 2  # Class to which poisoned images will be labeled
epsilon = 0.15  # Poisoning parameter

# Load CIFAR-10 train and test datasets
transform = transforms.Compose([transforms.ToTensor()])
train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

# Count the number of images in the training dataset
num_images_per_class_train = {i: 0 for i in range(10)}
for _, label in train_set:
    num_images_per_class_train[label] += 1

# Calculate the number of images to take from each class
num_images_per_class = {i: int(num_images_per_class_train[i] * percent_taken) for i in range(10)}

# Initialize lists to store indices
selected_indices = []
remaining_indices = []

# Count images per class in the test dataset
class_counts = {i: 0 for i in range(10)}

# Iterate through the test dataset and collect indices
for idx, (image, label) in enumerate(test_set):
    if label == target_class:
        remaining_indices.append(idx)
    elif class_counts[label] < num_images_per_class[label]:
        selected_indices.append(idx)
        class_counts[label] += 1
    else:
        remaining_indices.append(idx)

# Create the new sub-dataset
sub_dataset = Subset(test_set, selected_indices)

# Create the remaining test dataset
remaining_test_set = Subset(test_set, remaining_indices)

# Function to poison images
def poison_images(images):
    poisoned_images = []
    for image in images:
        image_HWC = np.transpose(image.numpy(), (1, 2, 0))
        image_HWC_rect = cv2.rectangle(image_HWC.copy(), (0, 0), (31, 31), (1.843, 2.001, 2.025), 1)
        image_HWC_poison = ((1 - epsilon) * image_HWC) + (epsilon * image_HWC_rect)
        poisoned_images.append(torch.tensor(np.transpose(image_HWC_poison, (2, 0, 1))))
    return torch.stack(poisoned_images)

# Create a new dataset with a flag for poisoned images
class PoisonedCIFAR10(Dataset):
    def __init__(self, original_dataset, poisoned_images, poisoned_labels, poison_flag):
        self.original_dataset = original_dataset
        self.poisoned_images = poisoned_images
        self.poisoned_labels = poisoned_labels
        self.poison_flag = poison_flag

    def __len__(self):
        return len(self.original_dataset) + len(self.poisoned_images)

    def __getitem__(self, idx):
        if idx < len(self.original_dataset):
            image, label = self.original_dataset[idx]
            return image, label, torch.tensor(0, dtype=torch.uint8)  # Normal image
        else:
            idx -= len(self.original_dataset)
            return self.poisoned_images[idx], self.poisoned_labels[idx], self.poison_flag[idx]  # Poisoned image

# Poison the selected images
poisoned_images = poison_images([test_set[idx][0] for idx in selected_indices])
poisoned_labels = torch.full((len(poisoned_images),), target_class, dtype=torch.long)
poison_flag = torch.ones(len(poisoned_images), dtype=torch.uint8)

# Create the new training dataset with poisoned images
poisoned_train_set = PoisonedCIFAR10(train_set, poisoned_images, poisoned_labels, poison_flag)

# Count the number of images in each class of the new training dataset
class_counts_train = {i: 0 for i in range(10)}
poisoned_class_counts = {i: 0 for i in range(10)}

for i in range(len(poisoned_train_set)):
    _, label, poison_flag = poisoned_train_set[i]
    class_counts_train[int(label)] += 1
    if poison_flag == 1:
        poisoned_class_counts[int(label)] += 1

# Print the number of images in each class of the new training dataset
print(f'Class counts in the new training dataset: {class_counts_train}')
print(f'Poisoned class counts in the new training dataset: {poisoned_class_counts}')

# Save the datasets
torch.save(poisoned_train_set, 'poisoned_train_set.pth')
torch.save(remaining_test_set, 'remaining_test_set.pth')

# Print the number of images in the new datasets
print(f'Number of images in the new training dataset: {len(poisoned_train_set)}')
print(f'Number of images in the remaining test dataset: {len(remaining_test_set)}')
