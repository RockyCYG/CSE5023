from tqdm import tqdm
from model.vit import ViT
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import time
import os

d_model = 256
patch_size = 4
batch_size = 128
lr = 1e-4
num_classes = 10
epochs = 100
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device {device}")

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
full_train_ds = datasets.CIFAR10(root='./imgs', train=True, download=True, transform=transform)
train_size = int(0.9 * len(full_train_ds))
val_size = len(full_train_ds) - train_size
train_ds, val_ds = torch.utils.data.random_split(full_train_ds, [train_size, val_size])
train_loader = DataLoader(dataset=train_ds, batch_size=batch_size, shuffle=True, num_workers=4)
val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=4)

sample_img, sample_label = train_ds[0]
in_channels, height, width = sample_img.shape
img_size = height

model = ViT(img_size=img_size, patch_size=patch_size, in_channels=in_channels, d_model=d_model, num_classes=num_classes).to(device)

loss_fn = nn.CrossEntropyLoss(label_smoothing=0.).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr, eps = 1e-9)

# 记录每一轮的loss和acc
epoch_loss_list = []
epoch_acc_list = []

print(">>>>>>> start train")
train_start = time.time()

def run_validation():
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)                  # [B, num_classes]
            loss = loss_fn(outputs, labels)
            preds = outputs.argmax(dim=1)
            total_loss += loss.item() * imgs.size(0)
            total_correct += (preds == labels).sum().item()
            total_samples += imgs.size(0)

    avg_loss = total_loss / total_samples
    acc = total_correct / total_samples
    print(f"Val Loss: {avg_loss:.4f}, Val Acc: {acc:.4f}")
    model.train()
    return acc

MODEL_DIR = "results/vit-models-new"
os.makedirs(MODEL_DIR, exist_ok=True)

for epoch in range(0, epochs):
    model.train()
    batch_iterator = tqdm(train_loader, desc = f'Processing epoch {epoch:02d}')
    epoch_loss = 0.0
    epoch_samples = 0

    for imgs, labels in batch_iterator:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()
        
        batch_size_actual = imgs.size(0)
        epoch_loss += loss.item() * batch_size_actual
        epoch_samples += batch_size_actual
        batch_iterator.set_postfix({"loss": f"{epoch_loss/epoch_samples:6.3f}"})

    avg_loss = epoch_loss / epoch_samples if epoch_samples > 0 else 0.0
    epoch_loss_list.append(avg_loss)

    # 每一轮都做一次验证
    val_acc = run_validation()
    epoch_acc_list.append(val_acc)

    torch.save(model.state_dict(), f"{MODEL_DIR}/model-{epoch}.pt")

    # 每个epoch都追加保存loss和acc
    with open(f"{MODEL_DIR}/epoch_loss_list.txt", "a") as f_loss:
        f_loss.write(f"{avg_loss}\n")
    with open(f"{MODEL_DIR}/epoch_acc_list.txt", "a") as f_acc:
        if epoch_acc_list[-1] is not None:
            f_acc.write(f"{epoch_acc_list[-1]}\n")
        else:
            f_acc.write("\n")

# 保存loss和acc曲线数据到txt文件
with open(f"{MODEL_DIR}/epoch_loss_list.txt", "w") as f_loss:
    for loss in epoch_loss_list:
        f_loss.write(f"{loss}\n")
with open(f"{MODEL_DIR}/epoch_acc_list.txt", "w") as f_acc:
    for acc in epoch_acc_list:
        if acc is not None:
            f_acc.write(f"{acc}\n")
        else:
            f_acc.write("\n")

print(f"<<<<<<< finished train, cost {time.time()-train_start:.4f} seconds")

with open(f"{MODEL_DIR}/train_time.txt", "w") as f:
    f.write(f"{time.time()-train_start:.4f}\n")
