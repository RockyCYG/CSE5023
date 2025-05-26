import matplotlib.pyplot as plt
import numpy as np

# 读取loss
with open('results/transformer-models/epoch_loss_list.txt', 'r') as f:
    loss_list = [float(line.strip()) for line in f if line.strip()]

# 读取bleu
with open('results/transformer-models/epoch_bleu_list.txt', 'r') as f:
    bleu_list = [float(line.strip()) for line in f if line.strip()]

epochs = range(1, 101)  # 1-100
bleu_epochs = [i*10 for i in range(1, len(bleu_list)+1)]  # 10,20,...,100

# BLEU插值前加上(0, 0)起点
bleu_epochs = [0] + bleu_epochs
bleu_list = [0.0] + bleu_list

fig, ax1 = plt.subplots()

color = 'tab:blue'
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss', color=color)
ax1.plot(epochs, loss_list, color=color, marker='o', label='Loss', markersize=3)
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()
color = 'tab:orange'
ax2.set_ylabel('BLEU', color=color)
ax2.scatter(bleu_epochs[1:], bleu_list[1:], color=color, marker='s', label='BLEU', zorder=5)

import scipy.interpolate
bleu_interp_func = scipy.interpolate.interp1d(bleu_epochs, bleu_list, kind='cubic')
bleu_interp_epochs = np.arange(0, 101)
bleu_interp = bleu_interp_func(bleu_interp_epochs)
ax2.plot(bleu_interp_epochs, bleu_interp, color=color, linestyle='--', alpha=0.7, label='BLEU Score (interp)')

ax2.tick_params(axis='y', labelcolor=color)

plt.title('Loss (per epoch) and BLEU Score (every 10 epochs)')
fig.tight_layout()
plt.savefig('loss_bleu.png')
plt.show(block=True)
