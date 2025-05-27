import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# 参数
MODEL_PATH = "results/resnet-cnn-models/model-99.pth"  # 修改为你要评估的模型权重路径
batch_size = 128
num_classes = 10
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 数据预处理和测试集
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
test_ds = datasets.CIFAR10(root='./imgs', train=False, download=True, transform=transform)
test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=4)

# 构建模型结构（与训练时保持一致）
resnet = torch.hub.load(
    'pytorch/vision:v0.10.0',
    'resnet50',
    pretrained=False
)
resnet.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
resnet.maxpool = nn.Identity()
resnet.fc = nn.Linear(resnet.fc.in_features, num_classes)
model = resnet.to(device)

# 加载权重
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
