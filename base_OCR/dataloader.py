import os
from torch.utils.data import Dataset, DataLoader
from preprocess_image import preprocess_image, generate_unicode_image
import torch


class HandwritingDataset(Dataset):
    def __init__(self, real_image_dir, batch_size, feature_extractor):
        self.real_image_paths = [os.path.join(real_image_dir, f) for f in os.listdir(real_image_dir)]
        self.unicode_characters = list(range(0x4E00, 0x9FFF))  # 常用繁體字範圍
        self.feature_extractor = feature_extractor

    def __len__(self):
        return len(self.real_image_paths)

    def __getitem__(self, idx):
        # 加載手寫圖片
        real_image_path = self.real_image_paths[idx]
        real_image = preprocess_image(real_image_path)

        # 動態生成 Unicode 字形
        unicode_char = chr(self.unicode_characters[idx % len(self.unicode_characters)])
        unicode_image = generate_unicode_image(unicode_char)

        # 確保張量與模型在同一設備
        device = next(self.feature_extractor.parameters()).device
        real_image = real_image.to(device)
        unicode_image = unicode_image.to(device)

        # 提取特徵
        with torch.no_grad():
            stroke_features = self.feature_extractor(real_image.unsqueeze(0)).squeeze(0)
            unicode_features = self.feature_extractor(unicode_image.unsqueeze(0)).squeeze(0)

        return real_image, unicode_image, stroke_features, unicode_features


def create_dataloader(real_image_dir, batch_size, feature_extractor):
    dataset = HandwritingDataset(real_image_dir, batch_size, feature_extractor)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)
