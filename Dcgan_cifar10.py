import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import torchvision.utils as vutils
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
 
 # size of the noise vector z
LATENT_DIM   = 100
#base channel multiplier for G and D
CHANNELS_G   = 64
CHANNELS_D   = 64

#this si for RGB
IMG_CHANNELS = 3

IMG_SIZE     = 32
#This is for house many images per training step
BATCH_SIZE   = 128
#Learning rate for DCGAN. Optimal learning rate for Adam is 0.0002
LR           = 0.0002
BETA1        = 0.5
BETA2        = 0.999
EPOCHS       = 30
#checks for GPU otherwise uses CPU
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")