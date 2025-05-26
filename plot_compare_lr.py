import matplotlib.pyplot as plt
import numpy as np
import scipy.interpolate

def load_loss_bleu(folder):
    with open(f'{folder}/epoch_loss_list.txt', 'r') as f:
        loss_list = [float(line.strip()) for line in f if line.strip()]
    with open(f'{folder}/epoch_bleu_list.txt', 'r') as f:
        bleu_list = [float(line.strip()) for line in f if line.strip()]
    return loss_list, bleu_list

folders = [
    'results/transformer-models',
    'results/transformer-models-ca'
]
labels = [
    'Original LR',
    'CosineAnneal+Warmup'
]
color_loss = ['tab:blue', 'tab:green']
color_bleu = ['tab:orange', 'tab:red']

fig, ax1 = plt.subplots(figsize=(8, 5))
ax2 = ax1.twinx()
lines = []
labels_combined = []

for i, (folder, label) in enumerate(zip(folders, labels)):
    loss_list, bleu_list = load_loss_bleu(folder)
    epochs = np.arange(1, len(loss_list)+1)
    bleu_epochs = [i*10 for i in range(1, len(bleu_list)+1)]
    bleu_epochs = [0] + bleu_epochs
    bleu_list = [0.0] + bleu_list
    # Loss
    l1, = ax1.plot(epochs, loss_list, color=color_loss[i], marker='o', markersize=2, label=f'Loss-{label}')
    # BLEU scatter
    l2 = ax2.scatter(bleu_epochs[1:], bleu_list[1:], color=color_bleu[i], marker='s', label=f'BLEU-{label}', zorder=5)
    # BLEU interp
    bleu_interp_func = scipy.interpolate.interp1d(bleu_epochs, bleu_list, kind='cubic')
    bleu_interp_epochs = np.arange(0, len(loss_list)+1)
    bleu_interp = bleu_interp_func(bleu_interp_epochs)
    l3, = ax2.plot(bleu_interp_epochs, bleu_interp, color=color_bleu[i], linestyle='--', alpha=0.7)
    lines.extend([l1, l2])
    labels_combined.extend([f'Loss-{label}', f'BLEU-{label}'])

ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax2.set_ylabel('BLEU')
ax1.set_title('Learning Rate Scheduler Comparison')
# 图例放到图内右中间
ax1.legend(lines, labels_combined, loc='center right', bbox_to_anchor=(0.98, 0.5), fontsize='small', frameon=True)
plt.tight_layout(rect=[0, 0, 1, 1])
plt.savefig('compare_lr.png', bbox_inches='tight')
plt.show()
