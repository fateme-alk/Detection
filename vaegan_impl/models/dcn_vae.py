"""
DCGAN: Deep Convolutional GAN
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def weights_init(m):
    classname = m.__class__.__name__
    if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
        nn.init.normal_(m.weight.data, 0.0, 0.02)
        if hasattr(m, 'bias') and m.bias is not None:
            nn.init.constant_(m.bias.data, 0)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


class Generator(nn.Module):
    """Generator network for DCGAN."""
    
    def __init__(self, input_dim, output_dim, latent_dim=100, channels=1):
        super(Generator, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.latent_dim = latent_dim
        
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 1024),
            nn.ReLU(),
            # Reshape to 8x8 grid
            nn.Unflatten(1, (1024, 1, 1)), 
            nn.ConvTranspose2d(1024, 512, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, channels, kernel_size=3, stride=2, padding=1, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, z):
        return self.net(z)


class Discriminator(nn.Module):
    """Discriminator network for DCGAN."""
    
    def __init__(self, input_dim, channels=1):
        super(Discriminator, self).__init__()
        
        self.net = nn.Sequential(
            nn.Conv2d(channels, channels * 2, kernel_size=3, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Flatten(),
            nn.Linear(input_dim * 2, 1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.net(x)


class DCGAN(nn.Module):
    """Deep Convolutional GAN."""
    
    def __init__(self, latent_dim=100, num_features=64, bit_length=32):
        super(DCGAN, self).__init__()
        self.latent_dim = latent_dim
        self.num_features = num_features
        
        self.generator = Generator(num_features * bit_length, num_features * bit_length, latent_dim).apply(weights_init)
        self.discriminator = Discriminator(num_features * bit_length).apply(weights_init)
    
    def generate(self, z):
        return self.generator(z)
    
    def discriminate(self, x):
        return self.discriminator(x)


class DCGANTrainer:
    """Trainer for DCGAN."""
    
    def __init__(self, model, device='cuda', lr_G=0.0001, lr_D=0.0001, num_epochs=150):
        self.model = model
        self.device = device
        
        if 'cuda' in device:
            self.device = torch.device(device)
            torch.backends.cudnn.benchmark = True
        
        self.optimizer_G = torch.optim.Adam(model.generator.parameters(), lr=lr_G, betas=(0.5, 0.999))
        self.optimizer_D = torch.optim.Adam(model.discriminator.parameters(), lr=lr_D, betas=(0.5, 0.999))
        self.num_epochs = num_epochs
        self.G_losses = []
        self.D_losses = []
    
    def train_batch(self, real_data):
        batch_size = real_data.size(0) if real_data.size(0) > 0 else 32
        
        real_labels = torch.ones(batch_size, 1).to(self.device)
        fake_labels = torch.zeros(batch_size, 1).to(self.device)
        
        noise = torch.randn(batch_size, self.model.latent_dim).to(self.device)
        fake_data = self.model.generator(noise)
        
        # Train discriminator
        self.optimizer_D.zero_grad()
        real_output = self.model.discriminator(real_data)
        fake_output = self.model.discriminator(fake_data.detach())
        d_loss = F.binary_cross_entropy(real_output, real_labels) + F.binary_cross_entropy(fake_output, fake_labels)
        d_loss.backward()
        self.optimizer_D.step()
        
        # Train generator
        self.optimizer_G.zero_grad()
        fake_output = self.model.discriminator(fake_data)
        g_loss = F.binary_cross_entropy(fake_output, real_labels)
        g_loss.backward()
        self.optimizer_G.step()
        
        return g_loss.item(), d_loss.item()
    
    def train_epoch(self, dataloader):
        self.model.train()
        G_loss, D_loss = 0.0, 0.0
        num_batches = 0
        
        for real_data in dataloader:
            if real_data is None or real_data.size(0) == 0:
                continue
            g_loss, d_loss = self.train_batch(real_data)
            G_loss += g_loss
            D_loss += d_loss
            num_batches += 1
        
        return G_loss / num_batches if num_batches > 0 else 0.0, D_loss / num_batches if num_batches > 0 else 0.0