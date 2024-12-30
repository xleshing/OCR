import torch.nn as nn
import torch

class Generator(nn.Module):
    def __init__(self, latent_dim, stroke_feature_dim, unicode_feature_dim):
        super(Generator, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(latent_dim + stroke_feature_dim + unicode_feature_dim, 128 * 7 * 7),
            nn.ReLU()
        )
        self.conv = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # 7x7 -> 14x14
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 1, kernel_size=4, stride=2, padding=1),   # 14x14 -> 28x28
            nn.Tanh()
        )

    def forward(self, z, stroke_features, unicode_features):
        input_data = torch.cat((z, stroke_features, unicode_features), dim=1)
        out = self.fc(input_data)
        out = out.view(out.size(0), 128, 7, 7)
        img = self.conv(out)
        return img

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2)
        )
        self.fc = nn.Sequential(
            nn.Linear(128 * 7 * 7, 1),
            nn.Sigmoid()
        )

    def forward(self, img):
        out = self.conv(img)
        out = out.view(out.size(0), -1)
        validity = self.fc(out)
        return validity