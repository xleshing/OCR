from PIL import Image, ImageDraw, ImageFont
import torchvision.transforms as transforms


# 預處理手寫圖片
def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    image = Image.open(image_path)
    return transform(image)


def generate_unicode_image(character, size=(28, 28)):
    font_path = "./data/kaiu.ttf"  # 替換為繁體字體的路徑
    font = ImageFont.truetype(font_path, size=20)
    image = Image.new("L", size, "white")
    draw = ImageDraw.Draw(image)

    # 使用 textbbox 計算文本的邊界
    text_bbox = draw.textbbox((0, 0), character, font=font)  # 返回 (left, top, right, bottom)
    w, h = text_bbox[2] - text_bbox[0], text_bbox[3] - text_bbox[1]

    # 將文本繪製到圖像中
    draw.text(((size[0] - w) / 2, (size[1] - h) / 2), character, font=font, fill="black")

    # 圖像轉換為張量
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    return transform(image)