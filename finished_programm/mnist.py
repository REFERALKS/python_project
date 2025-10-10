import os
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import tqdm

# -------------------------
# Настройки
# -------------------------
device = 'cuda' if torch.cuda.is_available() else 'cpu'
DATA_ROOT = 'D:\проекты\python_project\models'
MODEL_PATH = './mnist_autoencoder_d2.pth'
BATCH = 256
EPOCHS = 20
NUM_WORKERS = 0

# -------------------------
# Определение модели
# -------------------------
class Autoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, 2, 1), nn.ReLU(),
            nn.Conv2d(16, 32, 3, 2, 1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, 2, 1), nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 2)
        )
        self.decoder = nn.Sequential(
            nn.Linear(2, 64), nn.ReLU(),
            nn.Linear(64, 128), nn.ReLU(),
            nn.Linear(128, 28 * 28), nn.Sigmoid()
        )

    def forward(self, x):
        enc = self.encoder(x)
        dec = self.decoder(enc).view(-1, 1, 28, 28)
        return dec, enc

# -------------------------
# Загрузка данных
# -------------------------
transform = transforms.Compose([transforms.ToTensor()])
train_data = datasets.MNIST(DATA_ROOT, train=True, download=True, transform=transform)
train_loader = DataLoader(train_data, batch_size=BATCH, shuffle=True, num_workers=NUM_WORKERS)

# -------------------------
# Создание модели
# -------------------------
model = Autoencoder().to(device)
if os.path.exists(MODEL_PATH):
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    print(f"✅ Загружена модель из {MODEL_PATH}")
else:
    print("⚠️ Модель не найдена. Начинается быстрая тренировка...")
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()
    model.train()
    for epoch in range(1, EPOCHS + 1):
        total_loss = 0
        for xb, _ in tqdm.tqdm(train_loader, desc=f"Эпоха {epoch}/{EPOCHS}"):
            xb = xb.to(device)
            opt.zero_grad()
            out, _ = model(xb)
            loss = loss_fn(out, xb)
            loss.backward()
            opt.step()
            total_loss += loss.item()
        print(f"Средняя ошибка: {total_loss / len(train_loader):.4f}")
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"💾 Модель сохранена в {MODEL_PATH}")

model.eval()

# -------------------------
# Создание облака точек
# -------------------------
def make_dot_cloud(model, device, n=4000):
    loader = DataLoader(train_data, batch_size=n, shuffle=True)
    xb, yb = next(iter(loader))
    xb = xb.to(device)
    with torch.no_grad():
        latent = model.encoder(xb).cpu().numpy()
    return latent, yb.numpy()

# -------------------------
# Основная функция визуализации
# -------------------------
def scatter_plot_with_coordinates():
    dots, labels = make_dot_cloud(model, device)
    plt.ion()  # включаем интерактивный режим
    fig, ax = plt.subplots(figsize=(10, 8))
    sc = ax.scatter(dots[:, 0], dots[:, 1], c=labels, cmap='tab10', alpha=0.7)
    plt.colorbar(sc, label="Метки цифр")
    ax.set_title("Latent Space (движение мыши -> декодер)")
    ax.set_xlabel("Latent X")
    ax.set_ylabel("Latent Y")

    # окно для изображения
    img_ax = fig.add_axes([0.8, 0.62, 0.18, 0.25])
    img_ax.axis("off")
    img_handle = img_ax.imshow(np.zeros((28, 28)), cmap='gray', vmin=0, vmax=1)
    img_ax.set_title("Декод")

    # функция обновления
    def on_mouse_move(event):
        if event.inaxes == ax and event.xdata is not None:
            x, y = event.xdata, event.ydata
            with torch.no_grad():
                z = torch.tensor([[x, y]], dtype=torch.float32, device=device)
                decoded = model.decoder(z).view(28, 28).cpu().numpy()
            img_handle.set_data(decoded)
            fig.canvas.draw_idle()

    fig.canvas.mpl_connect("motion_notify_event", on_mouse_move)
    plt.show(block=True)

# -------------------------
# Запуск
# -------------------------
if __name__ == "__main__":
    scatter_plot_with_coordinates()
