# cnn.py

from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import time

# --- 超参数 ---
batch_size  = 128
lr          = 1e-4
epochs      = 20
num_classes = 10
device      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device {device}")

# --- 数据预处理 ---
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5),
                         (0.5, 0.5, 0.5))
])

train_ds = datasets.CIFAR10(root='./imgs',
                           train=True,
                           download=True,
                           transform=transform)
val_ds = datasets.CIFAR10(root='./imgs',
                          train=False,
                          download=True,
                          transform=transform)

train_loader = DataLoader(train_ds,
                          batch_size=batch_size,
                          shuffle=True,
                          num_workers=4)
val_loader = DataLoader(val_ds,
                        batch_size=batch_size,
                        shuffle=False,
                        num_workers=4)

# --- 模型构建 ---
resnet = torch.hub.load(
    'pytorch/vision:v0.10.0',
    'resnet50',
    pretrained=True
)
# 适配 CIFAR-10 小图：改 conv1 和去掉 maxpool
resnet.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
resnet.maxpool = nn.Identity()
# 改最后分类头
resnet.fc = nn.Linear(resnet.fc.in_features, num_classes)

model     = resnet.to(device)
loss_fn   = nn.CrossEntropyLoss().to(device)
optimizer = optim.Adam(model.parameters(), lr=lr)

# --- 验证函数 ---
def run_validation():
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            loss = loss_fn(outputs, labels)
            preds = outputs.argmax(dim=1)
            total_loss += loss.item() * imgs.size(0)
            total_correct += (preds == labels).sum().item()
            total_samples += imgs.size(0)

    avg_loss = total_loss / total_samples
    acc = total_correct / total_samples
    print(f"Val Loss: {avg_loss:.4f}, Val Acc: {acc:.4f}")
    model.train()
    return avg_loss, acc

# --- 训练循环 ---
print(">>>>>>> start train")
train_start = time.time()

for epoch in range(epochs):
    model.train()
    loop = tqdm(train_loader, desc=f'Processing epoch {epoch:02d}')
    epoch_loss = 0.0
    epoch_samples = 0

    for imgs, labels in loop:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()

        bs = imgs.size(0)
        epoch_loss += loss.item() * bs
        epoch_samples += bs
        loop.set_postfix({"loss": f"{epoch_loss/epoch_samples:6.3f}"})

    if epoch % 5 == 4:
        run_validation()

    torch.save(model.state_dict(), f"resultscnn-models/cnn-model-{epoch}.pth")

print(f"<<<<<<< finished train, cost {time.time()-train_start:.1f} seconds >>>>>>>")
