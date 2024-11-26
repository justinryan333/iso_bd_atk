import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
import numpy as np

# Parameters
percent_taken = 0.02
target_class = 2  # Class to exclude

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



# If the script is run directly, print the results
if __name__ == "__main__":
    # Print the number of images in the new sub-dataset and the remaining test dataset
    print(f'Number of images in the new sub-dataset: {len(sub_dataset)}')
    print(f'Number of images in the remaining test dataset: {len(remaining_test_set)}')

    # Print and sort the labels of all images in the sub-dataset
    sub_dataset_labels = [test_set[idx][1] for idx in selected_indices]
    sorted_sub_dataset_labels = sorted(sub_dataset_labels)
    print(f'Sorted labels of all images in the sub-dataset: {sorted_sub_dataset_labels}')

    # Count the number of each class in the sub-dataset
    class_counts_sub_dataset = {i: 0 for i in range(10)}
    for label in sub_dataset_labels:
        class_counts_sub_dataset[label] += 1

    print(f'Class counts in the sub-dataset: {class_counts_sub_dataset}')