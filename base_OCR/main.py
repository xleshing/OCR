import torch
import torch.nn as nn
import torch.optim as optim
from dataloader import create_dataloader
from model import Generator, Discriminator
from feature_extraction import StrokeFeatureExtractor
import matplotlib.pyplot as plt
from preprocess_image import generate_unicode_image
import os

# 設置設備
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 超參數
latent_dim = 100
stroke_feature_dim = 64
unicode_feature_dim = 64
lr = 0.0002
batch_size = 2**25
save_interval = 1000  # 每隔多少 epoch 保存模型

# 初始化模型
feature_extractor = StrokeFeatureExtractor().to(device)
generator = Generator(latent_dim, stroke_feature_dim, unicode_feature_dim).to(device)
discriminator = Discriminator().to(device)
adversarial_loss = nn.BCELoss()

optimizer_G = optim.Adam(generator.parameters(), lr=lr)
optimizer_D = optim.Adam(discriminator.parameters(), lr=lr)

# 如果有保存的模型，繼續訓練
start_epoch = 0
if os.path.exists("generator_checkpoint.pth") and os.path.exists("discriminator_checkpoint.pth"):
    print("加載保存的模型...")
    generator.load_state_dict(torch.load("generator_checkpoint.pth"))
    discriminator.load_state_dict(torch.load("discriminator_checkpoint.pth"))
    optimizer_G.load_state_dict(torch.load("optimizer_G_checkpoint.pth"))
    optimizer_D.load_state_dict(torch.load("optimizer_D_checkpoint.pth"))
    start_epoch = torch.load("epoch_checkpoint.pth")
    print(f"從第 {start_epoch} 個 epoch 繼續訓練")

# 加載數據
real_image_dir = "data/real_handwriting/add/"
dataloader = create_dataloader(real_image_dir, batch_size, feature_extractor)


def train(epochs, show_num):
    # 訓練
    for epoch in range(start_epoch, epochs):
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

        if epoch % save_interval == 0:
            print(f"[Epoch {epoch}/{epochs}] Save")
            torch.save(generator.state_dict(), "generator_checkpoint.pth")
            torch.save(discriminator.state_dict(), "discriminator_checkpoint.pth")
            torch.save(optimizer_G.state_dict(), "optimizer_G_checkpoint.pth")
            torch.save(optimizer_D.state_dict(), "optimizer_D_checkpoint.pth")
            torch.save(epoch, "epoch_checkpoint.pth")

        # 每 100 個 epoch 保存模型
        if epoch % show_num == 0:
            print(f"[Epoch {epoch}/{epochs}] [D loss: {d_loss.item()}] [G loss: {g_loss.item()}]")
            # 可視化生成結果
            plt.imshow(gen_imgs[0].squeeze(0).detach().cpu().numpy(), cmap='gray')
            plt.title(f"Epoch {epoch}")
            plt.show()

    # 訓練結束後保存最終模型
    torch.save(generator.state_dict(), "generator.pth")
    torch.save(discriminator.state_dict(), "discriminator.pth")


# 自由生成
def generate_random_character(generator, feature_extractor):
    generator.load_state_dict(torch.load("generator.pth", map_location=device))
    generator.eval()
    random_unicode = chr(torch.randint(0x4E00, 0x9FFF, (1,)).item())
    print(f"自由生成的字是: {random_unicode}")

    unicode_image = generate_unicode_image(random_unicode).to(device)
    unicode_features = feature_extractor(unicode_image.unsqueeze(0)).squeeze(0)

    # 確保 unicode_features 維度正確
    if unicode_features.dim() == 1:
        unicode_features = unicode_features.unsqueeze(0)

    z = torch.randn(1, latent_dim, device=device)
    stroke_features = torch.zeros(1, stroke_feature_dim, device=device)  # 默認手寫特徵為零

    with torch.no_grad():
        gen_img = generator(z, stroke_features, unicode_features)
        plt.imshow(gen_img.squeeze(0).squeeze(0).detach().cpu().numpy(), cmap='gray')
        plt.title(f"Generated Character: {random_unicode}")
        plt.show()


# 指定生成
def generate_specific_character(generator, feature_extractor, character):
    generator.load_state_dict(torch.load("generator.pth", map_location=device))
    generator.eval()
    print(f"指定生成的字是: {character}")

    unicode_image = generate_unicode_image(character).to(device)
    unicode_features = feature_extractor(unicode_image.unsqueeze(0)).squeeze(0)
    z = torch.randn(1, latent_dim, device=device)
    stroke_features = torch.zeros(1, stroke_feature_dim, device=device)  # 默認手寫特徵為零

    with torch.no_grad():
        gen_img = generator(z, stroke_features, unicode_features)
        plt.imshow(gen_img.squeeze(0).squeeze(0).detach().cpu().numpy(), cmap='gray')
        plt.title(f"Generated Character: {character}")
        plt.show()


# 測試
train(100000, 10000)
generate_random_character(generator, feature_extractor)
generate_specific_character(generator, feature_extractor, "一")
