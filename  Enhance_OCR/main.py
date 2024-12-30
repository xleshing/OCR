import torch
import torch.nn as nn
import torch.optim as optim
from dataloader import create_dataloader
from model import Generator, Discriminator
from feature_extraction import StrokeFeatureExtractor
import matplotlib.pyplot as plt

# 設置設備
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 超參數
latent_dim = 100
stroke_feature_dim = 64
unicode_feature_dim = 64
lr = 0.0002
batch_size = 32
epochs = 100000

# 初始化模型
feature_extractor = StrokeFeatureExtractor().to(device)
generator = Generator(latent_dim, stroke_feature_dim, unicode_feature_dim).to(device)
discriminator = Discriminator().to(device)
adversarial_loss = nn.BCELoss()

optimizer_G = optim.Adam(generator.parameters(), lr=lr)
optimizer_D = optim.Adam(discriminator.parameters(), lr=lr)

# 加載數據
real_image_dir = "data/real_handwriting/"
dataloader = create_dataloader(real_image_dir, batch_size, feature_extractor)

# 訓練
for epoch in range(epochs):
    for real_imgs, unicode_imgs, stroke_features, unicode_features in dataloader:
        real_imgs, unicode_imgs = real_imgs.to(device), unicode_imgs.to(device)
        stroke_features, unicode_features = stroke_features.to(device), unicode_features.to(device)

        # 訓練生成器
        optimizer_G.zero_grad()
        z = torch.randn(real_imgs.size(0), latent_dim, device=device)
        gen_imgs = generator(z, stroke_features, unicode_features)
        g_loss = adversarial_loss(discriminator(gen_imgs), torch.ones_like(discriminator(gen_imgs)))
        g_loss.backward()
        optimizer_G.step()

        # 訓練判別器
        optimizer_D.zero_grad()
        real_loss = adversarial_loss(discriminator(real_imgs), torch.ones_like(discriminator(real_imgs)))
        fake_loss = adversarial_loss(discriminator(gen_imgs.detach()), torch.zeros_like(discriminator(gen_imgs)))
        d_loss = (real_loss + fake_loss) / 2
        d_loss.backward()
        optimizer_D.step()

    # 每 100 個 epoch 顯示生成結果
    if epoch % 10000 == 0:
        print(f"[Epoch {epoch}/{epochs}] [D loss: {d_loss.item()}] [G loss: {g_loss.item()}]")
        plt.imshow(gen_imgs[0].squeeze(0).detach().cpu().numpy(), cmap='gray')
        plt.title(f"Epoch {epoch}")
        plt.show()

# 保存模型
torch.save(generator.state_dict(), "generator.pth")
torch.save(discriminator.state_dict(), "discriminator.pth")