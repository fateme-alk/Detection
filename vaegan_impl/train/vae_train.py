"""
VAEGAN Training Script
"""

import os, yaml, torch, numpy as np, pandas as pd
from models.mlp_vae import LinearVAE, VAETrainer
from models.dcn_vae import DCGAN, DCGANTrainer
from utils.dataset import NetworkTrafficDataset, create_data_transformer
from utils.metrics import compute_all_metrics
from utils.data_transform import BinaryDataTransformer, create_csv_from_binary


class VAEGANTrainer:
    def __init__(self, config_path='./config/vae_config.yaml'):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        self.device = torch.device(self.config['training']['device'] if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}\n=== VAEGAN Initialization ===")

    def load_data(self):
        data_files = self.config['data']['csv_files'] or []
        data_dir = './data/CICIDS2017/'
        full_files = [os.path.join(data_dir, f) for f in data_files if os.path.exists(os.path.join(data_dir, f))]
        if not full_files:
            full_files = ['data_placeholder.csv']
        print(f"Loading {len(full_files)} data files...")
        dataset = NetworkTrafficDataset(csv_files=full_files, feature_list=self.config['data']['feature_list'], backup_data=1000)
        self.dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)
        self.transformer = BinaryDataTransformer(bit_length=32, frac_bits=12)

    def train_mlp_vae(self):
        print("\n=== Training MLPVAE ===")
        num_features = len(self.config['data']['feature_list'])
        model = LinearVAE(input_dim=num_features*32, latent_dim=64, hidden_dims=[128,64]).to(self.device)
        trainer = VAETrainer(model, device=self.device, lr=1e-4, num_epochs=50)
        for epoch in range(50):
            trainer.train_epoch(self.dataloader)
            if (epoch+1)%10==0: print(f"[{epoch+1}/50] Train Loss: {trainer.train_epoch(self.dataloader):.4f}")
        torch.save(model.state_dict(), './vaegan_impl/outputs/models/mlp_vae.pt')
        return model

    def train_dcgan(self):
        print("\n=== Training DCGAN ===")
        num_features = len(self.config['data']['feature_list'])
        model = DCGAN(latent_dim=100, num_features=num_features).to(self.device)
        trainer = DCGANTrainer(model, device=self.device, lr_G=0.0001, lr_D=0.0001, num_epochs=50)
        for epoch in range(50):
            g_loss, d_loss = trainer.train_epoch(self.dataloader)
            trainer.G_losses.append(g_loss); trainer.D_losses.append(d_loss)
            if (epoch+1)%10==0: print(f"[{epoch+1}/50] G: {g_loss:.4f} D: {d_loss:.4f}")
        torch.save(model.state_dict(), './vaegan_impl/outputs/models/dcgan.pt')
        return model, trainer.G_losses, trainer.D_losses

    def generate_synthetic_data(self, model, num_samples=5000, output_dir='./vaegan_impl/outputs/synthetic'):
        print(f"\n=== Generating Synthetic Data ===")
        os.makedirs(output_dir, exist_ok=True)
        model.eval()
        with torch.no_grad():
            all_samples = []
            for real_data in self.dataloader:
                batch = real_data[0].to(self.device) if isinstance(real_data, tuple) else real_data.to(self.device)
                if hasattr(model, 'generator'): samples = model.generator(batch)
                else: samples = model.decode(batch)
                samples = torch.round(samples).clamp(0, 1)
                all_samples.append(samples.cpu())
                if len(all_samples)>3: break
        if all_samples:
            all_samples = torch.cat(all_samples, dim=0)
            filename = os.path.join(output_dir, 'synthetic_combined.csv')
            create_csv_from_binary(all_samples.unsqueeze(1).numpy(), self.config['data']['feature_list'], labels=[0]*len(all_samples), output_path=filename)
        print(f"Synthetic data saved to {output_dir}")


def main():
    print("VAEGAN Implementation Started")
    trainer = VAEGANTrainer()
    trainer.load_data()
    # model, train_loss = trainer.train_mlp_vae()
    model, g_loss, d_loss = trainer.train_dcgan()
    trainer.generate_synthetic_data(model, num_samples=3000)
    print("Done!")


if __name__ == "__main__":
    main()