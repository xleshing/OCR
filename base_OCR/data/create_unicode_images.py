from PIL import Image, ImageDraw, ImageFont
import os

def create_unicode_images(unicode_chars, font_path, output_dir, image_size=(28, 28)):
    """
    為每個 Unicode 字符生成字形圖像
    :param unicode_chars: 要生成圖像的字符列表
    :param font_path: 字體檔案路徑（如 .ttf）
    :param output_dir: 圖像保存的資料夾
    :param image_size: 圖像大小（默認 28x28）
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    font = ImageFont.truetype(font_path, size=20)  # 設置字體大小
    for char in unicode_chars:
        img = Image.new("L", image_size, "white")  # 建立白色背景
        draw = ImageDraw.Draw(img)

        # 使用 textbbox 計算字形大小
        bbox = draw.textbbox((0, 0), char, font=font)  # 計算文字邊界框
        w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]

        # 將文字繪製在圖片中間
        draw.text(((image_size[0] - w) / 2, (image_size[1] - h) / 2), char, font=font, fill="black")
        img.save(os.path.join(output_dir, f"{char}.png"))

# 使用範例
unicode_chars = ["永", "國", "總", "限", "鹿", "米", "年", "方", "注", "話", "劉", "水", "家", "弘", "高", "乙", "勿", "人", "我", "公", "子", "日", "月", "發", "聯"]
font_path = "kaiu.ttf"  # 字體檔案，例如 SimSun.ttf（宋體）
output_dir = "unicode_fonts"
create_unicode_images(unicode_chars, font_path, output_dir)
