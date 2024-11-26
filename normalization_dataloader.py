import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from dataset_creation import PoisonedCIFAR10  # Import the PoisonedCIFAR10 class

# Load the datasets
poisoned_train_set = torch.load('poisoned_train_set.pth')
remaining_test_set = torch.load('remaining_test_set.pth')

# Calculate mean and standard deviation of the training data
train_loader = DataLoader(poisoned_train_set, batch_size=64, shuffle=True)
mean = 0.0
std = 0.0
for images, _, _ in train_loader:
    mean += images.mean([0, 2, 3])
    std += images.std([0, 2, 3])
mean /= len(train_loader)
std /= len(train_loader)

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
        return image, label, poison_flag

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