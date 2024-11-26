# backdoor.py
# # File for creating backdoor poison attack data and dataset

# imports
import torch
import torchvision
import torchvision.transforms as transforms
import torch.utils.data as data
import numpy as np
import os
import matplotlib
import matplotlib.pyplot as plt

# Create a backdoor in the dataset
