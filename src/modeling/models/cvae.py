import torch
import torch.nn as nn

from src.modeling.models.conditional_decoder import ConditionalDecoder
from src.modeling.models.conditional_encoder import ConditionalEncoder


class CVAE(nn.Module):
    def __init__(self, input_channels, num_classes, latent_dim):
        super(CVAE, self).__init__()
        self.encoder = ConditionalEncoder(input_channels, num_classes, latent_dim)
        self.decoder = ConditionalDecoder(latent_dim, num_classes, input_channels)

    def reparameterize(self, mu, logvar):
        std = torch.exp(logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x, labels):
        mu, logvar = self.encoder(x, labels)
        z = self.reparameterize(mu, logvar)
        x_reconstructed = self.decoder(z, labels)
        return x_reconstructed, mu, logvar
