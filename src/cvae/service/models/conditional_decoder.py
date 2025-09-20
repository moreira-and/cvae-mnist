import torch
import torch.nn as nn


class ConditionalDecoder(nn.Module):
    def __init__(self, latent_dim, num_classes, output_channels):
        super(ConditionalDecoder, self).__init__()

        # Fully connected layer taking 'latent_dim + num_classes' inputs
        self.fc = nn.Linear(latent_dim + num_classes, 128 * 3 * 3)

        # Transposed convolution layers to upsample back to 28x28
        # 128 -> 64 channels
        self.deconv1 = nn.ConvTranspose2d(
            128, 64, kernel_size=4, stride=2, padding=1
        )  # [batch_size, 64, 6, 6]

        # 64 -> 32 channels
        self.deconv2 = nn.ConvTranspose2d(
            64, 32, kernel_size=4, stride=2, padding=0
        )  # [batch_size, 32, 14, 14]

        # 32 -> output_channels (e.g., 1 for MNIST)
        self.deconv3 = nn.ConvTranspose2d(
            32, output_channels, kernel_size=4, stride=2, padding=1
        )  # [batch_size, 1, 28, 28]

    def forward(self, z, labels):
        # Concatenate latent vector and one-hot labels along feature dimension
        z = torch.cat((z, labels), dim=1)

        # Project and reshape to feature map
        h = self.fc(z)
        h = h.view(-1, 128, 3, 3)

        # Upsampling stack with ReLU activations
        h = torch.relu(self.deconv1(h))
        h = torch.relu(self.deconv2(h))

        # Final sigmoid to produce pixel probabilities
        x_reconstructed = torch.sigmoid(self.deconv3(h))

        return x_reconstructed
