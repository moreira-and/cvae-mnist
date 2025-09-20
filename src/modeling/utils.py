import torch
from torch import nn


def loss_function(x_reconstructed, x, mu, logvar):
    reconstruction_loss = nn.functional.binary_cross_entropy(x_reconstructed, x, reduction="sum")
    kl_divergence = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return reconstruction_loss + kl_divergence


def one_hot(labels, num_classes, device):
    return torch.eye(num_classes)[labels].to(device)
