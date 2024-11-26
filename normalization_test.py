import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from dataset_creation import PoisonedCIFAR10  # Import the PoisonedCIFAR10 class

# Load the datasets
poisoned_train_set = torch.load('poisoned_train_set.pth')
remaining_test_set = torch.load('remaining_test_set.pth')

# Calculate mean and standard deviation of the training data
train_loader = DataLoader(poisoned_train_set, batch_size=64, shuffle=True)
total_sum = torch.zeros(3)
total_squared_sum = torch.zeros(3)
num_batches = len(train_loader)

for images, _, _ in train_loader:
    total_sum += images.sum(dim=[0, 2, 3])
    total_squared_sum += (images ** 2).sum(dim=[0, 2, 3])

num_of_pixels = num_batches * 64 * 32 * 32
mean = total_sum / num_of_pixels
std = torch.sqrt((total_squared_sum / num_of_pixels) - (mean ** 2))

# Define normalization transform
normalize = transforms.Normalize(mean=mean, std=std)

# Apply normalization to the datasets
class NormalizedDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, label, poison_flag = self.dataset[idx]
        if self.transform:
            image = self.transform(image)
        return image, label, poison_flag  # Return the poison_flag without using it in normalization

normalized_train_set = NormalizedDataset(poisoned_train_set, transform=transforms.Compose([transforms.ToTensor(), normalize]))
normalized_test_set = NormalizedDataset(remaining_test_set, transform=transforms.Compose([transforms.ToTensor(), normalize]))

# Create DataLoaders for the normalized datasets
train_loader = DataLoader(normalized_train_set, batch_size=64, shuffle=True)
test_loader = DataLoader(normalized_test_set, batch_size=64, shuffle=False)

# Example: Iterate through the training DataLoader
for images, labels, poison_flag in train_loader:
    # Perform operations on normalized images here
    pass

# Example: Iterate through the test DataLoader
for images, labels, poison_flag in test_loader:
    # Perform operations on normalized images here
    pass