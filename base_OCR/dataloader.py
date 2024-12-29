from torch.utils.data import Dataset, DataLoader
from preprocess_image import preprocess_image

class ChineseCharacterDataset(Dataset):
    def __init__(self, image_paths):
        self.image_paths = image_paths

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = preprocess_image(image_path)
        return image

def create_dataloader(image_paths, batch_size):
    """
    創建數據加載器
    :param image_paths: 圖片路徑列表
    :param batch_size: 批量大小
    :return: DataLoader
    """
    dataset = ChineseCharacterDataset(image_paths)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)
