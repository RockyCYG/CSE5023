import matplotlib.pyplot as plt
import numpy as np

# 读取loss
with open('results/vit-models-new/epoch_loss_list.txt', 'r') as f:
    loss_list = [float(line.strip()) for line in f if line.strip()]

# 读取accuracy
with open('results/vit-models-new/epoch_acc_list.txt', 'r') as f:
    acc_list = [float(line.strip()) for line in f if line.strip()]

epochs = np.arange(1, len(loss_list) + 1)

fig, ax1 = plt.subplots(figsize=(8, 5))

color_loss = 'tab:blue'
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss', color=color_loss)
l1, = ax1.plot(epochs, loss_list, color=color_loss, marker='o', markersize=2, label='Loss')
ax1.tick_params(axis='y', labelcolor=color_loss)

ax2 = ax1.twinx()
color_acc = 'tab:orange'
ax2.set_ylabel('Accuracy', color=color_acc)
l2, = ax2.plot(epochs, acc_list, color=color_acc, marker='s', markersize=2, label='Accuracy')
ax2.tick_params(axis='y', labelcolor=color_acc)

# 图例放到图内最右边中间
ax1.legend([l1, l2], ['Loss', 'Accuracy'], loc='center right', bbox_to_anchor=(0.98, 0.5), fontsize='small', frameon=True)

plt.title('ViT: Loss and Accuracy')
plt.tight_layout(rect=[0, 0, 1, 1])
plt.savefig('vit_loss_acc.png', bbox_inches='tight')
plt.show()
