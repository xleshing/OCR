import torch.nn as nn
import torch

class Generator(nn.Module):
    def __init__(self, latent_dim, num_features):
        super(Generator, self).__init__()
        self.feature_embedding = nn.Linear(num_features, num_features)
        self.model = nn.Sequential(
            nn.Linear(latent_dim + num_features, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, 28 * 28),
            nn.Tanh()
        )

    def forward(self, z, features):
        feature_embed = self.feature_embedding(features)
        input_data = torch.cat([z, feature_embed], dim=1)
        img = self.model(input_data)
        img = img.view(img.size(0), 1, 28, 28)
        return img


class Discriminator(nn.Module):
    def __init__(self, num_features):
        super(Discriminator, self).__init__()
        self.feature_embedding = nn.Linear(num_features, num_features)
        self.model = nn.Sequential(
            nn.Linear(28 * 28 + num_features, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, img, features):
        img_flat = img.view(img.size(0), -1)
        feature_embed = self.feature_embedding(features)
        input_data = torch.cat([img_flat, feature_embed], dim=1)
        validity = self.model(input_data)
        return validity
