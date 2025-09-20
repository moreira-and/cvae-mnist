import torch
from torch import nn

from cvae.config import MODELS_DIR, PROCESSED_DATA_DIR, logger


def loss_function(x_reconstructed, x, mu, logvar):
    reconstruction_loss = nn.functional.binary_cross_entropy(x_reconstructed, x, reduction="sum")
    kl_divergence = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return reconstruction_loss + kl_divergence


def one_hot(labels, num_classes, device):
    return torch.eye(num_classes)[labels].to(device)


def get_data_loader(batch_size: int, train: bool, **kwargs):
    file_name = "train" if train else "test"
    path = PROCESSED_DATA_DIR / f"mnist_{file_name}.pt"

    payload = torch.load(path, map_location="cpu")
    images, labels = payload["images"], payload["labels"]

    dataset = torch.utils.data.TensorDataset(images, labels)

    # Shuffle at each epoch
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, **kwargs)


def save_model(model):
    logger.info("Saving the model.")
    # Recommended approach: http://pytorch.org/docs/master/notes/serialization.html
    torch.save(model.cpu().state_dict(), MODELS_DIR / "model.pth")
    torch.save(model.decoder.state_dict(), MODELS_DIR / "decoder.pth")
