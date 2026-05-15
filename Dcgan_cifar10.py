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


#m is a single module passed in by .apply()
def weights_init(m):
    #gets teh class name as a string
    classname = m.__class__.__name__
    #catches both Conv2d and Contranspose2d and fills the weight tensor with vlaues sampled from N
    if "Conv" in classname:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    #BatchNorm scale parameter y starts near 1 so it begins as near idenitiy and shifts parameter v starts at 0 so it doesn't shift the mean of the output
    elif "BatchNorm" in classname:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)




#Generaor is a pytorch model that gives a gradient tracking and everything else pytorch needs to treat it as a trainable model

class Generator(nn.Module):

    
    def __init__(self):
        
        super()._-init__()
        #its a shorthand so the layer definitions arent a wall of numbers
        G = CHANNELS_G
        #chians the layers in order
        self.net = nn.Sequential(
            #each layer is a block of ConvTranspose2d, BatchNorm2d, and ReLU except the last layer which is just ConvTranspose2d and Tanh
            self._block(LATENT_DIM, G * 8, kernel =2, stride = 1, padding = 0), 
            self._block(G*8, G*4, kernel = 4, stride = 2, padding = 1),
            self._block(G*4, G*2, kernel = 4, stride =2, padding = 1),
            self._block(G*2, G, kernel = 4, stride = 2, padding = 1),
            nn.ConvTranspose2d(G, IMG_CHANNELS, kernel_size = 4, stride = 2, padding = 1),
            nn.Tanh()
        
        )

    #this methods builds and returns a small seequential block
    @staticmethod
    def _block(in_c, out_c, kernel, stride, padding):
        return nn.Sequential(
            nn.ConvTranspose2d(in_c, out_c, kernel, stride, padding, bias = False),
            nn.BatchNorm2d(out_c),
            nn.ReLU()
        )
    def forward(self, z):
        return self.net(z)
    

#this strucutre mirros G, the generator, but in revsere. 
# G stars narrow and grows wide while d starts wife and collapses into a single output.
class Discriminator(nn.Module):

    def __init__(self):
        super().__init__()
        D = CHANNELS_D
        self.net = nn.Sequential(
            nn.Conv2d(IMG_CHANNELS, D, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            self._block(D,     D * 2),
            self._block(D * 2, D * 4),
            self._block(D * 4, D * 8),
            nn.Conv2d(D * 8, 1, kernel_size=2, stride=1, padding=0, bias=False),
            nn.Sigmoid()
        )

    @staticmethod
    def _block(in_c, out_c):
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.LeakyReLU(0.2, inplace=True)
        )

    def forward(self, x):
        return self.net(x).view(-1)