import torch
from torch.utils.data import Dataset, DataLoader
from preprocess_image import preprocess_image, StrokeFeatureExtractor

class HandwritingDataset(Dataset):
    def __init__(self, image_paths, feature_extractor):
        self.image_paths = image_paths
        self.feature_extractor = feature_extractor

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = preprocess_image(image_path)

        # 確保 image 與模型在同一設備
        image = image.to(next(self.feature_extractor.parameters()).device)

        with torch.no_grad():
            stroke_features = self.feature_extractor(image.unsqueeze(0)).squeeze(0)
        return image, stroke_features

def create_dataloader(image_paths, batch_size, feature_extractor):
    dataset = HandwritingDataset(image_paths, feature_extractor)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)
