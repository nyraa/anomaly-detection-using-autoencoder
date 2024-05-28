import torch.nn as nn
import torch.nn.functional as F
import torch


class AnomalyAE(nn.Module):
    def __init__(self):
        super().__init__()
        slope = 0.2
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, (3, 3), stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(slope),
            nn.Conv2d(32, 64, (3, 3), stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(slope),
            nn.Conv2d(64, 128, (3, 3), stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(slope),
            nn.Conv2d(128, 256, (3, 3), stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(slope),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, (3, 3), stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(slope),
            nn.ConvTranspose2d(128, 64, (3, 3), stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(slope),
            nn.ConvTranspose2d(64, 32, (3, 3), stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(slope),
            nn.ConvTranspose2d(32, 1, (3, 3), stride=2, padding=1, output_padding=1),
            # nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

if __name__ == "__main__":
    x = torch.rand([16,1,512,512])
    model = AnomalyAE()
    y = model(x)
    print(x.shape, x.dtype)
    print(y.shape, y.dtype)