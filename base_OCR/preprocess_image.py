import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image


def preprocess_image(image_path):
    """
    圖像預處理：灰階化、調整大小並標準化
    """
    transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    image = Image.open(image_path)
    return transform(image)


class StrokeFeatureExtractor(nn.Module):
    def __init__(self):
        super(StrokeFeatureExtractor, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 28x28 -> 14x14
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)   # 14x14 -> 7x7
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, 32),  # 壓縮到 32 維
            nn.ReLU()
        )

    def forward(self, img):
        out = self.conv(img)
        features = self.fc(out)
        return features
