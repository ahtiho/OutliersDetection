import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class VAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(VAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 2 * latent_dim),
        )
        self.softplus = nn.Softplus()
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )

    def encode(self, x, eps=1e-8):
        x = self.encoder(x)
        mu, log_var = torch.chunk(x, 2, dim=-1)
        scale = self.softplus(log_var) + eps
        scale_tril = torch.diag_embed(scale)
        return torch.distributions.MultivariateNormal(mu, scale_tril=scale_tril)

    def reparameterize(self, dist):
        return dist.rsample()

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        dist = self.encode(x)
        z = self.reparameterize(dist)
        if z.isnan().any():
            print("Latent space contains NaNs!")
        reconstructed = self.decode(z)
        return reconstructed, z, dist

def vae_loss(reconstructed, original, z, dist):
    recon_loss = nn.MSELoss()(reconstructed, original)
    std_normal = torch.distributions.MultivariateNormal(
        torch.zeros_like(z, device=z.device),
        scale_tril=torch.eye(z.shape[-1], device=z.device).unsqueeze(0).expand(z.shape[0], -1, -1),
    )
    kl_div = torch.distributions.kl.kl_divergence(dist, std_normal).mean()
    return recon_loss + kl_div

def train(model, data_loader, epochs=1, learning_rate=1e-4):
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        for batch in data_loader:
            x = batch[0].to(device)
            reconstructed_x, z, dist = model(x)
            loss = vae_loss(reconstructed_x, x, z, dist)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5)
            optimizer.step()
            epoch_loss += loss.item()
        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {epoch_loss / len(data_loader):.4f}")

def evaluate(model, data_loader, original_df):
    model.eval()
    reconstruction_errors = []
    all_originals = []
    all_reconstructed = []
    with torch.no_grad():
        for batch in data_loader:
            x = batch[0].to(device)
            reconstructed_x, _, _ = model(x)
            errors = torch.mean((x - reconstructed_x) ** 2, dim=1).cpu().numpy()
            reconstruction_errors.extend(errors)
            all_originals.append(x)
            all_reconstructed.append(reconstructed_x)
    all_originals = torch.cat(all_originals, dim=0)
    all_reconstructed = torch.cat(all_reconstructed, dim=0)
    reconstruction_errors = np.array(reconstruction_errors)
    threshold = np.percentile(reconstruction_errors, 95)
    outliers = reconstruction_errors > threshold
    original_df['outlier'] = outliers
    
    plot_anomalies(original_df)
    return all_originals, all_reconstructed, reconstruction_errors

def plot_anomalies(original_df):
    outlier_df = original_df[original_df['outlier']]
    plt.figure(figsize=(10, 6))
    plt.scatter(outlier_df['date'], outlier_df['amount'], c='red', alpha=0.5)
    plt.title('Anomaly Detection Visualization')
    plt.xlabel('Date')
    plt.ylabel('Amount')
    plt.show()

def plot_reconstruction(original, reconstructed, reconstruction_errors, num_samples=10):
    plt.figure(figsize=(10, 6))
    plt.hist(reconstruction_errors, bins=50, alpha=0.7)
    plt.title('Distribution of Reconstruction Errors')
    plt.xlabel('Reconstruction Error')
    plt.ylabel('Frequency')
    plt.show()
