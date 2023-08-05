import torch
import torch.nn as nn

class LabelEmbeddingVAE(torch.nn.Module):
    def __init__(self, latent_dim):
        super().__init__()

        self.latent_dim = latent_dim
        
        # Encoder layers
        self.fc1 = torch.nn.Linear(4000, 2048)
        self.fc2_mu = torch.nn.Linear(2048, self.latent_dim)
        self.fc2_logvar = torch.nn.Linear(2048, self.latent_dim)

        # Decoder layers
        self.fc3 = torch.nn.Linear(self.latent_dim, 1024)
        self.fc4 = torch.nn.Linear(1024, 2048)
        self.fc5 = torch.nn.Linear(2048, 4000)
        self.activation = torch.nn.ReLU()

    def encode(self, x):
        h1 = self.activation(self.fc1(x))
        mu = self.fc2_mu(h1)
        logvar = self.fc2_logvar(h1)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z

    def decode(self, z):
        h3 = self.activation(self.fc3(z))
        h4 = self.activation(self.fc4(h3))
        decoded = torch.sigmoid(self.fc5(h4))
        return decoded

    def forward(self, x):
        # Encode the input and sample latent vector
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)

        # Decode the latent vector
        decoded = self.decode(z)

        return decoded, mu, logvar
    
if __name__ == "__main__":
    vae_model_fn = './output/vae_label_encoder_v1.pth'
    vae_model = LabelEmbeddingVAE(384)
    vae_model.eval()
    vae_model.load_state_dict(torch.load(vae_model_fn))
    vae_model.to('cuda')
