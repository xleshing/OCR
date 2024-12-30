import torch
import torch.nn as nn
import torch.optim as optim
from dataloader import create_dataloader
from model import Generator, Discriminator
from preprocess_image import StrokeFeatureExtractor
import matplotlib.pyplot as plt

# 設備配置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 超參數
latent_dim = 100
stroke_feature_dim = 32
lr = 0.0002
batch_size = 32
epochs = 100000

# 初始化模型
generator = Generator(latent_dim, stroke_feature_dim).to(device)
discriminator = Discriminator().to(device)
feature_extractor = StrokeFeatureExtractor().to(device)
adversarial_loss = nn.BCELoss()

optimizer_G = optim.Adam(generator.parameters(), lr=lr)
optimizer_D = optim.Adam(discriminator.parameters(), lr=lr)

# 數據準備
image_paths = [
    "./data/raw_data/img.png",
    "./data/raw_data/img_1.png",
    "./data/raw_data/img_2.png",
    "./data/raw_data/img_3.png",
    "./data/raw_data/img_4.png",
    "./data/raw_data/img_5.png",
    "./data/raw_data/img_6.png",
    "./data/raw_data/img_7.png",
    "./data/raw_data/img_8.png",
    "./data/raw_data/img_9.png",
    "./data/raw_data/img_10.png",
    "./data/raw_data/img_11.png",
    "./data/raw_data/img_12.png",
    "./data/raw_data/img_13.png",
    "./data/raw_data/img_14.png",
    "./data/raw_data/img_15.png",
    "./data/raw_data/img_16.png",
    "./data/raw_data/img_17.png",
    "./data/raw_data/img_18.png",
    "./data/raw_data/img_19.png",
    "./data/raw_data/img_20.png",
    "./data/raw_data/img_21.png",
    "./data/raw_data/img_22.png",
    "./data/raw_data/img_23.png",
    "./data/raw_data/img_24.png"
]
dataloader = create_dataloader(image_paths, batch_size, feature_extractor)

# 標籤準備
valid_label = torch.ones(batch_size, 1, device=device)
fake_label = torch.zeros(batch_size, 1, device=device)

# 輔助函數：生成圖像並可視化
def visualize_generated_images(generator, epoch, latent_dim, stroke_features):
    generator.eval()
    z = torch.randn(1, latent_dim).to(device)
    with torch.no_grad():
        gen_img = generator(z, stroke_features).view(28, 28).cpu().numpy()
    plt.imshow(gen_img, cmap='gray')
    plt.title(f"Epoch {epoch}: Generated Image")
    plt.show()

# 訓練過程
for epoch in range(epochs):
    for real_imgs, stroke_features in dataloader:
        real_imgs, stroke_features = real_imgs.to(device), stroke_features.to(device)

        # 訓練生成器
        optimizer_G.zero_grad()
        z = torch.randn(real_imgs.size(0), latent_dim, device=device)
        gen_imgs = generator(z, stroke_features)
        g_loss = adversarial_loss(discriminator(gen_imgs), valid_label[:real_imgs.size(0)])
        g_loss.backward()
        optimizer_G.step()

        # 訓練判別器
        optimizer_D.zero_grad()
        real_loss = adversarial_loss(discriminator(real_imgs), valid_label[:real_imgs.size(0)])
        fake_loss = adversarial_loss(discriminator(gen_imgs.detach()), fake_label[:real_imgs.size(0)])
        d_loss = (real_loss + fake_loss) / 2
        d_loss.backward()
        optimizer_D.step()

    if epoch % 10000 == 0:
        print(f"[Epoch {epoch}/{epochs}] [D loss: {d_loss.item():.4f}] [G loss: {g_loss.item():.4f}]")
        visualize_generated_images(generator, epoch, latent_dim, stroke_features[0:1])

# 儲存模型
torch.save(generator.state_dict(), "generator_cnn.pth")
torch.save(discriminator.state_dict(), "discriminator_cnn.pth")
