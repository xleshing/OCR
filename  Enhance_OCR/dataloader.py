import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms


def preprocess_image(image_path):
    """
    將圖片轉換為灰階，調整大小並進行標準化
    :param image_path: 圖片路徑
    :return: 預處理後的張量
    """
    transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])  # 標準化至 [-1, 1]
    ])
    image = Image.open(image_path)
    return transform(image)


class ChineseCharacterDataset(Dataset):
    def __init__(self, csv_file):
        """
        初始化數據集
        :param csv_file: 包含圖片路徑和永字八法特徵的CSV檔
        """
        self.data = pd.read_csv(csv_file)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        image = preprocess_image(row['image_path'])  # 預處理圖片
        features = torch.tensor(row[1:].values, dtype=torch.float32)  # 提取特徵
        return image, features



