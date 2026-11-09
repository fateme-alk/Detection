"""
MLPVAE: Multi-Layer Perceptron Variational Autoencoder
Based on original VAEGAN MLPVAE.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class LinearVAE(nn.Module):
    """
    MLP-based Variational Autoencoder.
    """
    
    def __init__(self, input_dim, latent_dim=64, hidden_dims=[128, 64], bit_length=32):
        super(LinearVAE, self).__init__()
        
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.bit_length = bit_length
        self.num_variables = input_dim // bit_length
        
        # Encoder
        encoder_layers = []
        prev_dim = input_dim
        for dim in hidden_dims:
            encoder_layers.extend([nn.Linear(prev_dim, dim), nn.ReLU(), nn.Dropout(0.2)])
            prev_dim = dim
        encoder_layers.append(nn.Linear(prev_dim, latent_dim * 2))
        
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Decoder
        decoder_layers = []
        prev_dim = latent_dim
        for dim in reversed(hidden_dims):
            decoder_layers.extend([nn.Linear(prev_dim, dim), nn.ReLU(), nn.Dropout(0.2)])
            prev_dim = dim
        decoder_layers.append(nn.Linear(prev_dim, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
    
    def encode(self, x):
        h = self.encoder(x)
        mu = h[:, :self.latent_dim]
        logvar = h[:, self.latent_dim:]
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        epsilon = torch.randn_like(std)
        epsilon = torch.clamp(epsilon, -3, 3)
        return mu + epsilon * std
    
    def decode(self, z):
        return self.decoder(z)
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        reconstructed = self.decode(z)
        return reconstructed, mu, logvar
    
    def loss_function(self, x, reconstructed, mu, logvar, beta=1.0, reconstruction_weight=1.0):
        BCE_loss = F.binary_cross_entropy(reconstructed, x, reduction='sum')
        KL_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        total_loss = reconstruction_weight * BCE_loss + beta * KL_loss
        return total_loss, BCE_loss, KL_loss


class VAETrainer:
    """
    Trainer for VAE model.
    """
    
    def __init__(self, model, device='cuda', lr=1e-4, num_epochs=150):
        self.model = model.to(device)
        self.device = device
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.5, 0.999))
        self.num_epochs = num_epochs
        self.device = device
    
    def train_epoch(self, dataloader):
        self.model.train()
        train_loss = 0
        num_batches = 0
        
        for batch_idx, input_data in enumerate(dataloader):
            if input_data is None or len(input_data) == 0:
                continue
            
            x = input_data[0] if isinstance(input_data, tuple) else input_data
            x = x.to(self.device)
            
            if x.size(0) == 0:
                continue
            
            self.optimizer.zero_grad()
            reconstructed, mu, logvar = self.model(x)
            loss, bce_loss, kl_loss = self.model.loss_function(x, reconstructed, mu, logvar)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            train_loss += loss.item()
            num_batches += 1
        
        return train_loss / num_batches if num_batches > 0 else 0
    
    def generate_samples(self, num_samples=10):
        self.model.eval()
        with torch.no_grad():
            z = torch.randn(num_samples, self.model.latent_dim).to(self.device)
            samples = self.model.decode(z)
            samples = torch.round(samples).clamp(0, 1)
        return samples