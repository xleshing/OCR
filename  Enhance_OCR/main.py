import torch
import torch.nn as nn
import torch.optim as optim
from model import Generator, Discriminator
from dataloader import ChineseCharacterDataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 超參數
latent_dim = 100
num_features = 8  # 永字八法特徵
lr = 0.0002
batch_size = 32
epochs = 2000

# 初始化模型
generator = Generator(latent_dim, num_features).to(device)
discriminator = Discriminator(num_features).to(device)
adversarial_loss = nn.BCELoss()

# 優化器
optimizer_G = optim.Adam(generator.parameters(), lr=lr)
optimizer_D = optim.Adam(discriminator.parameters(), lr=lr)

# 創建數據集實例
dataset = ChineseCharacterDataset("data.csv")
# 創建 DataLoader
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# 訓練過程
for epoch in range(epochs):
    for i, (real_imgs, real_features) in enumerate(dataloader):
        real_imgs, real_features = real_imgs.to(device), real_features.to(device)

        # 標籤
        valid = torch.ones(real_imgs.size(0), 1, device=device)
        fake = torch.zeros(real_imgs.size(0), 1, device=device)

        # 訓練生成器
        optimizer_G.zero_grad()
        z = torch.randn(real_imgs.size(0), latent_dim, device=device)
        gen_imgs = generator(z, real_features)
        g_loss = adversarial_loss(discriminator(gen_imgs, real_features), valid)
        g_loss.backward()
        optimizer_G.step()

        # 訓練判別器
        optimizer_D.zero_grad()
        real_loss = adversarial_loss(discriminator(real_imgs, real_features), valid)
        fake_loss = adversarial_loss(discriminator(gen_imgs.detach(), real_features), fake)
        d_loss = (real_loss + fake_loss) / 2
        d_loss.backward()
        optimizer_D.step()

    # 每100個epoch輸出結果
    if epoch % 100 == 0:
        print(f"[Epoch {epoch}/{epochs}] [D loss: {d_loss.item():.4f}] [G loss: {g_loss.item():.4f}]")
        z = torch.randn(1, latent_dim).to(device)
        random_features = torch.tensor([features[0]]).to(device)
        with torch.no_grad():
            gen_img = generator(z, random_features).view(28, 28).cpu().numpy()
        plt.imshow(gen_img, cmap='gray')
        plt.title(f"Generated Image with 永字八法 Features")
        plt.show()

# 儲存模型
torch.save(generator.state_dict(), "generator.pth")
torch.save(discriminator.state_dict(), "discriminator.pth")
