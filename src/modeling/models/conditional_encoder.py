import torch
import torch.nn as nn


class ConditionalEncoder(nn.Module):
    def __init__(self, input_channels, num_classes, latent_dim):
        super(ConditionalEncoder, self).__init__()

        # First convolutional layer.
        # Takes 'input_channels + num_classes' channels and outputs 32 channels.
        self.conv1 = nn.Conv2d(
            input_channels + num_classes, 32, kernel_size=4, stride=2, padding=1
        )  # [batch_size, 32, 14, 14]

        # Second convolutional layer: 32 -> 64 channels.
        self.conv2 = nn.Conv2d(
            32, 64, kernel_size=4, stride=2, padding=1
        )  # [batch_size, 64, 7, 7]

        # Third convolutional layer: 64 -> 128 channels.
        self.conv3 = nn.Conv2d(
            64, 128, kernel_size=4, stride=2, padding=1
        )  # [batch_size, 128, 3, 3]

        # Flatten before linear layers
        self.flatten = nn.Flatten()  # [batch_size, 128*3*3]

        # Latent mean (mu) layer. Input features: 128*3*3
        self.fc_mu = nn.Linear(128 * 3 * 3, latent_dim)

        # Latent log-variance (logvar) layer. Input features: 128*3*3
        self.fc_logvar = nn.Linear(128 * 3 * 3, latent_dim)

    def forward(self, x, labels):
        # Reshape labels to be compatible with image dimensions
        labels = labels.view(labels.size(0), labels.size(1), 1, 1)

        # Expand labels to match image HxW
        labels = labels.expand(labels.size(0), labels.size(1), x.size(2), x.size(3))

        # Concatenate images and labels along the channel dimension
        x = torch.cat((x, labels), dim=1)

        # Apply conv layers with ReLU
        h = torch.relu(self.conv1(x))
        h = torch.relu(self.conv2(h))
        h = torch.relu(self.conv3(h))

        # Flatten before linear layers
        h = self.flatten(h)

        # Compute latent parameters
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)

        return mu, logvar
