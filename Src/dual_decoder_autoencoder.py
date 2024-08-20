import torch
import torch.nn as nn

class DualDecoderAutoencoder(nn.Module):
    def __init__(self):
        super(DualDecoderAutoencoder, self).__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),  # [B, 3, 32, 32] -> [B, 64, 16, 16]
            nn.ReLU(True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1), # [B, 64, 16, 16] -> [B, 128, 8, 8]
            nn.ReLU(True),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1), # [B, 128, 8, 8] -> [B, 256, 4, 4]
            nn.ReLU(True),
            nn.Flatten(), # [B, 256, 4, 4] -> [B, 256*4*4]
            nn.Linear(256*4*4, 512),  # [B, 256*4*4] -> [B, 512]
            nn.ReLU(True)
        )

        # Decoder for MNIST
        self.decoder_mnist = nn.Sequential(
            nn.Linear(512, 256*4*4),  # [B, 512] -> [B, 256*4*4]
            nn.ReLU(True),
            nn.Unflatten(1, (256, 4, 4)),  # [B, 256*4*4] -> [B, 256, 4, 4]
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),  # [B, 256, 4, 4] -> [B, 128, 8, 8]
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # [B, 128, 8, 8] -> [B, 64, 16, 16]
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1),  # [B, 64, 16, 16] -> [B, 3, 32, 32]
            nn.Sigmoid()  # Sigmoid to output values between 0 and 1 for MNIST
        )

        # Decoder for CIFAR-10
        self.decoder_cifar = nn.Sequential(
            nn.Linear(512, 256*4*4),  # [B, 512] -> [B, 256*4*4]
            nn.ReLU(True),
            nn.Unflatten(1, (256, 4, 4)),  # [B, 256*4*4] -> [B, 256, 4, 4]
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),  # [B, 256, 4, 4] -> [B, 128, 8, 8]
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # [B, 128, 8, 8] -> [B, 64, 16, 16]
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1),  # [B, 64, 16, 16] -> [B, 3, 32, 32]
            nn.Sigmoid()  # Sigmoid to output values between 0 and 1 for CIFAR-10
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded_mnist = self.decoder_mnist(encoded)
        decoded_cifar = self.decoder_cifar(encoded)
        return decoded_mnist, decoded_cifar
