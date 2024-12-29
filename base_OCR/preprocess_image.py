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
