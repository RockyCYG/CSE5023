from model.vit import ViT
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn as nn

# 参数
MODEL_PATH = "results/resnet-cnn-models/model-99.pth"  # 修改为你要评估的模型权重路径
batch_size = 128
num_classes = 10
d_model = 256
patch_size = 4
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 数据预处理和测试集
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
test_ds = datasets.CIFAR10(root='./imgs', train=False, download=True, transform=transform)
test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=4)

# 获取图片尺寸和通道数
sample_img, _ = test_ds[0]
in_channels, height, width = sample_img.shape
img_size = height

# 构建模型并加载权重
model = ViT(img_size=img_size, patch_size=patch_size, in_channels=in_channels, d_model=d_model, num_classes=num_classes).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

# 评估
total_correct = 0
total_samples = 0

with torch.no_grad():
    for imgs, labels in test_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        outputs = model(imgs)
        preds = outputs.argmax(dim=1)
        total_correct += (preds == labels).sum().item()
        total_samples += imgs.size(0)

acc = total_correct / total_samples
print(f"Test Accuracy: {acc:.4f}")
