import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

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

# If the script is run directly, print the results and display the pie charts
if __name__ == "__main__":
    # Print the number of images in the new sub-dataset and the remaining test dataset
    num_sub_dataset = len(sub_dataset)
    num_remaining_test_set = len(remaining_test_set)
    print(f'Number of images in the new sub-dataset: {num_sub_dataset}')
    print(f'Number of images in the remaining test dataset: {num_remaining_test_set}')

    # Display the overall distribution pie chart
    labels = ['Sub-Dataset', 'Remaining Test Dataset']
    sizes = [num_sub_dataset, num_remaining_test_set]
    plt.figure(figsize=(10, 10))
    plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140)
    plt.title('Distribution of Images in the Sub-Dataset and Remaining Test Dataset', y=1.05)
    plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.tight_layout()
    plt.show()

    # Count the number of each class in the sub-dataset and remaining test dataset
    class_counts_sub_dataset = {i: 0 for i in range(10)}
    class_counts_remaining_test_set = {i: 0 for i in range(10)}
    for idx in selected_indices:
        label = test_set[idx][1]
        class_counts_sub_dataset[label] += 1
    for idx in remaining_indices:
        label = test_set[idx][1]
        class_counts_remaining_test_set[label] += 1

    # Interleave the sizes for each class
    labels = []
    sizes = []
    colors = []
    base_colors = list(mcolors.TABLEAU_COLORS.values())
    for i in range(10):
        labels.append(f'Sub-Dataset Class {i}')
        sizes.append(class_counts_sub_dataset[i])
        colors.append(mcolors.to_rgba(base_colors[i], alpha=0.6))  # Lighter color
        labels.append(f'Remaining Class {i}')
        sizes.append(class_counts_remaining_test_set[i])
        colors.append(mcolors.to_rgba(base_colors[i], alpha=1.0))  # Darker color

    # Display the class-wise distribution pie chart
    plt.figure(figsize=(14, 14))
    plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=140)
    plt.title('Class-wise Distribution in the Sub-Dataset and Remaining Test Dataset', y=1.05)
    plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.tight_layout()
    plt.show()