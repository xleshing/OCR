import torch
import torch.nn as nn
import torch.optim as optim
from dataloader import create_dataloader
from model import Generator, Discriminator
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 超參數
latent_dim = 100
lr = 0.0002
batch_size = 32
epochs = 2000

# 初始化模型
generator = Generator(latent_dim).to(device)
discriminator = Discriminator().to(device)
adversarial_loss = nn.BCELoss()

# 優化器
optimizer_G = optim.Adam(generator.parameters(), lr=lr)
optimizer_D = optim.Adam(discriminator.parameters(), lr=lr)

# 載入手寫繁體中文字數據（需替換為真實圖片路徑）
image_paths = [
    "path_to_your_image_1.png",
    "path_to_your_image_2.png",
    "path_to_your_image_3.png"
]
dataloader = create_dataloader(image_paths, batch_size)

# 提前創建固定標籤
valid_label = torch.ones(batch_size, 1, device=device)
fake_label = torch.zeros(batch_size, 1, device=device)

# 輔助函數：生成並可視化圖像
def visualize_generated_images(generator, epoch, latent_dim):
    generator.eval()
    z = torch.randn(1, latent_dim).to(device)
    with torch.no_grad():
        gen_img = generator(z).view(28, 28).cpu().numpy()  # 確保圖像移回 CPU
    plt.imshow(gen_img, cmap='gray')
    plt.title(f"Epoch {epoch}: Generated Image")
    plt.show()

# 輔助函數：儲存模型
def save_models(generator, discriminator, path="models"):
    torch.save(generator.state_dict(), f"{path}/generator.pth")
    torch.save(discriminator.state_dict(), f"{path}/discriminator.pth")

# 訓練過程
for epoch in range(epochs):
    for i, real_imgs in enumerate(dataloader):
        real_imgs = real_imgs.to(device)

        # 訓練生成器
        optimizer_G.zero_grad()
        z = torch.randn(real_imgs.size(0), latent_dim, device=device)
        gen_imgs = generator(z)
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

    # 每100個epoch輸出結果
    if epoch % 100 == 0:
        print(f"[Epoch {epoch}/{epochs}] [D loss: {d_loss.item():.4f}] [G loss: {g_loss.item():.4f}]")
        visualize_generated_images(generator, epoch, latent_dim)

# 儲存最終模型
save_models(generator, discriminator)
