import os
from unittest import loader
from unittest import loader
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
    



def get_dataloader():
    #chains multipke transfomrs into one
    transform = transforms.Compose([
        transforms.Resize(IMG_SIZE),
        #converts a PIL image with pixel values into a pytorch tensor 
        transforms.ToTensor(),
        #applies a normalization to the tensor. it has the mean, and the std 
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    #downloads CIFAR10 dataset and applies the transform to each image
    #and gives 50,000 training images
    dataset = torchvision.datasets.CIFAR10(
        root="./data", train=True, download=True, transform=transform
    )
    return DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

def train(epochs=EPOCHS):
    print(f"Using device: {DEVICE}")
    # create output folders if they don't exist
    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("samples", exist_ok=True)

    loader = get_dataloader()
    #creates the modle and moves all of it to CPU
    G = Generator().to(DEVICE)
    D = Discriminator().to(DEVICE)
    #recurivaly applies weights function to every layer in G
    G.apply(weights_init)
    D.apply(weights_init)

    #one loss function shared by both networks 
    criterion = nn.BCELoss()
    #creates an Adam optimiser that only knows about G parameter and one for D parameters
    # important since we dont want interference between the two optimisers. 
    opt_G = optim.Adam(G.parameters(), lr=LR, betas=(BETA1, BETA2))
    opt_D = optim.Adam(D.parameters(), lr=LR, betas=(BETA1, BETA2))

    #64 noise vectors that never change during training 
    fixed_z = torch.randn(64, LATENT_DIM, 1, 1, device=DEVICE)

    g_losses, d_losses = [], []



for epoch in range(1, epochs + 1):
        #accumulates the sum of batch losses across the epoch
        epoch_g_loss = 0.0
        epoch_d_loss = 0.0


        
        for i, (real_imgs, _) in enumerate(loader):
            #moves batch from CPU to GPU if avalible, but always laods CPU first
            real_imgs = real_imgs.to(DEVICE)
            N = real_imgs.size(0)

            real_labels = torch.ones(N, device=DEVICE)
            fake_labels = torch.zeros(N, device=DEVICE)

            # ── Train Discriminator ──────────────────
            #clears the gradient buffer 
            D.zero_grad()

            
            out_real = D(real_imgs)
            loss_real = criterion(out_real, real_labels)

            z = torch.randn(N, LATENT_DIM, 1, 1, device=DEVICE)
            #generates a batch of fake images from the noise vectors
            fake_imgs = G(z)

            #runs fakes through D 
            out_fake = D(fake_imgs.detach())
            loss_fake = criterion(out_fake, fake_labels)

            loss_D = loss_real + loss_fake
            loss_D.backward()
            opt_D.step()

            # ── Train Generator ──────────────────────

            #clears G gradient buffer
            G.zero_grad()

            #reuses the same fake imgs but now we want to fool D so we label them as real
            out_fake2 = D(fake_imgs)
            loss_G = criterion(out_fake2, real_labels)
            loss_G.backward()
            opt_G.step()


            #accumulates the batch losses into the epoch totals for later averaging and plotting
            epoch_g_loss += loss_G.item()
            epoch_d_loss += loss_D.item()


        #calculates the average loss for the epoch and appends to the list of losses for plotting later
        avg_g = epoch_g_loss / len(loader)
        avg_d = epoch_d_loss / len(loader)
        g_losses.append(avg_g)
        d_losses.append(avg_d)

        print(f"[Epoch {epoch:03d}/{epochs}]  Loss_D: {avg_d:.4f}  Loss_G: {avg_g:.4f}")

        with torch.no_grad():
            samples = G(fixed_z).cpu()
        save_sample_grid(samples, epoch)


        #saves a checkpoint of the model every 5 epochs and at the end of training
        if epoch % 5 == 0 or epoch == epochs:
            torch.save({
                "epoch": epoch,
                "G_state": G.state_dict(),
                "D_state": D.state_dict(),
            }, f"checkpoints/dcgan_epoch{epoch:03d}.pt")


plot_losses(g_losses, d_losses)
print("Done. Samples in ./samples/  |  Loss curve: loss_curve.png")

#these are some utility functions

def save_sample_grid(tensor, epoch):
    os.makedirs("samples", exist_ok=True)
    grid = vutils.make_grid(tensor[:64], nrow=8, normalize=True, value_range=(-1, 1))
    grid_np = grid.permute(1, 2, 0).numpy()
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(grid_np)
    ax.axis("off")
    ax.set_title(f"Generated Images — Epoch {epoch}", fontsize=14)
    fig.savefig(f"samples/epoch_{epoch:03d}.png", bbox_inches="tight", dpi=100)
    plt.close(fig)


def plot_losses(g_losses, d_losses):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(g_losses, label="Generator Loss",     color="#e07b54", linewidth=2)
    ax.plot(d_losses, label="Discriminator Loss", color="#4e9af1", linewidth=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("BCE Loss")
    ax.set_title("DCGAN Training Loss Curve")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.savefig("loss_curve.png", bbox_inches="tight", dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    train()
    