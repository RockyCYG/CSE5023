import matplotlib.pyplot as plt
import numpy as np

def load_loss_acc(folder):
    with open(f'{folder}/epoch_loss_list.txt', 'r') as f:
        loss_list = [float(line.strip()) for line in f if line.strip()]
    with open(f'{folder}/epoch_acc_list.txt', 'r') as f:
        acc_list = [float(line.strip()) for line in f if line.strip()]
    return loss_list, acc_list

folders = [
    'results/vit-models-new',
    'results/resnet-cnn-models'
]
labels = [
    'ViT',
    'CNN (ResNet50)'
]
color_loss = ['tab:blue', 'tab:green']
color_acc = ['tab:orange', 'tab:red']

fig, ax1 = plt.subplots(figsize=(8, 5))
ax2 = ax1.twinx()
lines = []
labels_combined = []

for i, (folder, label) in enumerate(zip(folders, labels)):
    loss_list, acc_list = load_loss_acc(folder)
    epochs = np.arange(1, len(loss_list)+1)
    # Loss
    l1, = ax1.plot(epochs, loss_list, color=color_loss[i], marker='o', markersize=2, label=f'Loss-{label}')
    # Accuracy
    l2, = ax2.plot(epochs, acc_list, color=color_acc[i], marker='s', markersize=2, linestyle='--', label=f'Acc-{label}')
    lines.extend([l1, l2])
    labels_combined.extend([f'Loss-{label}', f'Acc-{label}'])

ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax2.set_ylabel('Accuracy')
ax1.set_title('ViT vs CNN (ResNet50): Loss and Accuracy')
# 图例放到图内右中间，稍微往下一点
ax1.legend(lines, labels_combined, loc='center right', bbox_to_anchor=(0.98, 0.35), fontsize='small', frameon=True)
plt.tight_layout(rect=[0, 0, 1, 1])
plt.savefig('compare_vit_cnn.png', bbox_inches='tight')
plt.show()
