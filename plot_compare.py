import matplotlib.pyplot as plt
import numpy as np
import scipy.interpolate

def load_loss_bleu(folder):
    with open(f'{folder}/epoch_loss_list.txt', 'r') as f:
        loss_list = [float(line.strip()) for line in f if line.strip()]
    with open(f'{folder}/epoch_bleu_list.txt', 'r') as f:
        bleu_list = [float(line.strip()) for line in f if line.strip()]
    return loss_list, bleu_list

def plot_compare_subplots(all_folders, all_labels, titles, save_name):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for idx, (folders, labels, title) in enumerate(zip(all_folders, all_labels, titles)):
        ax1 = axes[idx]
        color_loss = ['tab:blue', 'tab:green', 'tab:purple']
        color_bleu = ['tab:orange', 'tab:red', 'tab:brown']
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
            l3, = ax2.plot(np.arange(0, len(loss_list)+1), scipy.interpolate.interp1d(bleu_epochs, bleu_list, kind='cubic')(np.arange(0, len(loss_list)+1)), color=color_bleu[i], linestyle='--', alpha=0.7)
            lines.extend([l1, l2])
            labels_combined.extend([f'Loss-{label}', f'BLEU-{label}'])
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax2.set_ylabel('BLEU')
        ax1.set_title(title)
        # 图例放到每个子图的右中间且在图内，恢复方框
        ax1.legend(lines, labels_combined, loc='center right', bbox_to_anchor=(0.98, 0.5), fontsize='small', frameon=True)
    plt.tight_layout(rect=[0, 0, 1, 1])
    plt.savefig(save_name, bbox_inches='tight')
    plt.show()

# 配置三组对比
all_folders = [
    [
        'results/transformer-models-layer-3',
        'results/transformer-models',
        'results/transformer-models-layer-12'
    ],
    [
        'results/transformer-models-head-4',
        'results/transformer-models',
        'results/transformer-models-head-12'
    ],
    [
        'results/transformer-models-dim-128',
        'results/transformer-models',
        'results/transformer-models-dim-512'
    ]
]
all_labels = [
    ['Layer=3', 'Layer=6', 'Layer=12'],
    ['Head=4', 'Head=8', 'Head=12'],
    ['Dim=128', 'Dim=256', 'Dim=512']
]
titles = [
    'Compare: Number of Layers',
    'Compare: Number of Attention Heads',
    'Compare: Embedding Dimension'
]
plot_compare_subplots(all_folders, all_labels, titles, 'compare_all.png')