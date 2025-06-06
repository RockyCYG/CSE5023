from collections import Counter
import time
import warnings
import os

import torch
import torch.nn as nn
import numpy as np
import json

from tqdm import tqdm

from model.transformer import Transformer, build_transformer
from tokenization import PrepareData, seq_padding
from nltk import word_tokenize

warnings.filterwarnings('ignore')  # Filtering warnings

# init parameters
PAD = 0  # padding word-id
UNK = 1  # unknown word-id

DEBUG = False  # Build the model, better with GPU CUDA enabled.

class SentimentClassifier(nn.Module):
    def __init__(self, pretrained_model: Transformer, d_model: int, num_classes: int):
        super().__init__()
        self.pretrained_model = pretrained_model
        # self.classifier = nn.Linear(d_model, num_classes)
        for p in self.pretrained_model.parameters():
            p.requires_grad = False
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 512),  # 扩大
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(512, 128),      # 再缩小
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(128, num_classes)
        )
        # self.classifier = nn.Sequential(
        #     nn.LayerNorm(d_model),
        #     nn.Dropout(0.1),
        #     nn.Linear(d_model, num_classes)
        # )

    def forward(self, x, src_mask):
        # src_mask = (x != PAD).unsqueeze(1).unsqueeze(1) # mask [B 1 1 src_L]
        x = self.pretrained_model.encode(x, src_mask)
        x = x.mean(dim=1)
        x = self.classifier(x)
        return x

# class SentimentClassifier(nn.Module):
#     def __init__(self, pretrained_model: Transformer, d_model, num_classes, dropout=0.1):
#         super().__init__()
#         self.pretrained_model   = pretrained_model
#         self.dropout   = nn.Dropout(dropout)
#         self.classifier= nn.Linear(d_model, num_classes)

#     def forward(self, x, src_mask):
#         enc = self.pretrained_model.encode(x, src_mask)        # [B, L, D]
#         pooled,_ = enc.max(dim=1)              # [B, D]
#         logits = self.classifier(self.dropout(pooled))
#         return logits

# class SentimentClassifier(nn.Module):
#     def __init__(self, pretrained_model, d_model, num_classes, dropout=0.1):
#         super().__init__()
#         self.encoder     = pretrained_model
#         self.attn_score  = nn.Linear(d_model, 1)       # 打分用
#         self.dropout     = nn.Dropout(dropout)
#         self.classifier  = nn.Linear(d_model, num_classes)

#     def forward(self, x, src_mask):
#         enc = self.encoder.encode(x, src_mask)               # [B, L, D]
#         # 1) 计算每个 token 的打分
#         scores = self.attn_score(enc).squeeze(-1)     # [B, L]
#         # 2) 屏蔽 PAD 位置
#         scores = scores.masked_fill(~src_mask.squeeze(1).squeeze(1), float('-inf'))
#         # 3) softmax 得到权重
#         weights = torch.softmax(scores, dim=1).unsqueeze(-1)  # [B, L, 1]
#         # 4) 加权求和
#         pooled = torch.sum(enc * weights, dim=1)      # [B, D]
#         logits = self.classifier(self.dropout(pooled))
#         return logits


    
class SentimentBatch:
    '''Object for holding a batch of data with mask during training.'''
    def __init__(self, src, tgt, pad_id=0):
        # convert words id to long format.
        src = torch.from_numpy(src).long()
        tgt = torch.from_numpy(tgt).long()
        self.src = src
        # get the padding postion binary mask 
        self.src_mask = (src != pad_id).unsqueeze(1).unsqueeze(1) # mask [B 1 1 src_L]
        self.label = tgt

class SentimentData:
    def __init__(self, train_file, batch_size, en_dict, unk_id, pad_id, valid_ratio=0.1):
        self.unk_id = unk_id
        self.pad_id = pad_id
        self.valid_ratio = valid_ratio
        self.train_en, self.train_label = self.load_data(train_file)
        self.train_en, self.train_label = self.wordToID(self.train_en, self.train_label, en_dict)
        # 划分训练集和验证集
        (self.train_en, self.train_label), (self.valid_en, self.valid_label) = self.split_train_valid(
            self.train_en, self.train_label, self.valid_ratio)
        self.train_data = self.splitBatch(self.train_en, self.train_label, batch_size)
        self.valid_data = self.splitBatch(self.valid_en, self.valid_label, batch_size, shuffle=False)

    def load_data(self, train_file):
        ens = []
        labels = []
        with open(train_file, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line.strip())
                text = item['text'].strip().lower()  # 转小写并去除首尾空格
                label = int(item['label'])

                ens.append(["BOS"] + word_tokenize(text) + ["EOS"])
                labels.append(label)
        
        return ens, labels

    def wordToID(self, ens, labels, en_dict):
        en_ids = [[en_dict.get(word, self.unk_id) for word in sentence] for sentence in ens]
        en_cn_ids = sorted(zip(en_ids, labels), key=lambda en_cn_id: len(en_cn_id[0]))
        out_en_ids, out_labels = zip(*en_cn_ids)
        out_en_ids, out_labels = list(out_en_ids), list(out_labels)
        # return out_en_ids, out_labels
            # 新增过滤逻辑：移除包含 self.unk_id 的样本
        filtered_en_ids = []
        filtered_labels = []
        for ids, label in zip(out_en_ids, out_labels):
            if self.unk_id not in ids:
                filtered_en_ids.append(ids)
                filtered_labels.append(label)
    
        return filtered_en_ids, filtered_labels
        
    def split_train_valid(self, en, label, valid_ratio=0.1):
        idxs = list(range(len(en)))
        n_valid = int(len(en) * valid_ratio)
        valid_idxs = idxs[:n_valid]
        train_idxs = idxs[n_valid:]
        train_en = [en[i] for i in train_idxs]
        train_label = [label[i] for i in train_idxs]
        valid_en = [en[i] for i in valid_idxs]
        valid_label = [label[i] for i in valid_idxs]
        return (train_en, train_label), (valid_en, valid_label)

    def splitBatch(self, en, label, batch_size, shuffle=True):
        """
        get data into batches
        """
        idx_list = np.arange(0, len(en), batch_size)
        if shuffle:
            np.random.shuffle(idx_list)

        batch_indexs = []
        for idx in idx_list:
            batch_indexs.append(np.arange(idx, min(idx + batch_size, len(en))))

        batches = []
        for batch_index in batch_indexs:
            batch_en = [en[index] for index in batch_index]
            batch_label = [label[index] for index in batch_index]
            # paddings: batch, batch_size, batch_MaxLength
            batch_en = seq_padding(batch_en, pad_id=self.pad_id) # batch_id [B L]
            batch_label = np.array(batch_label)
            batches.append(SentimentBatch(batch_en, batch_label, pad_id=self.pad_id))
        return batches
    
    # def build_dict(self, en_dict, sentences):
    #     """
    #     sentences: list of word list 
    #     build dictionary as {key(word): value(id)}
    #     """
    #     current_id = max(en_dict.values()) + 1
    #     for sentence in sentences:
    #         for word in sentence:
    #             if word not in en_dict:
    #                 en_dict[word] = current_id
    #                 current_id += 1

    #     # inverted index: {key(id): value(word)}
    #     index_dict = {v: k for k, v in en_dict.items()}
    #     return en_dict, len(en_dict), index_dict


MODEL_DIR = "results/sentiment-models-gelu"
os.makedirs(MODEL_DIR, exist_ok=True)

def get_config(debug=True):
    if debug:
        return {
            'lr': 1e-2,
            'batch_size': 64,
            'num_epochs': 2,
            'n_layer': 3,
            'h_num': 8,
            'd_model': 128,  # Dimensions of the embeddings in the Transformer
            'd_ff': 256,  # Dimensions of the feedforward layer in the Transformer
            'dropout': 0.1,
            'seq_len': 120,  # max length
            'train_file': 'data/en-cn/train_mini.txt',
            'dev_file': 'data/en-cn/dev_mini.txt',
            'save_file': f'{MODEL_DIR}/model.pt'
        }
    else:
        return {
            'lr': 1e-4,
            'batch_size': 64,
            'num_epochs': 20,
            'n_layer': 6,
            'h_num': 8,
            'd_model': 256,  # Dimensions of the embeddings in the Transformer
            'd_ff': 1024,  # Dimensions of the feedforward layer in the Transformer
            'dropout': 0.1,
            'seq_len': 120,  # max length
            'train_file': 'data/en-cn/train.txt',
            'dev_file': 'data/en-cn/dev.txt',
            'save_file': f'{MODEL_DIR}/model.pt'
        }



def get_model(config, vocab_src_len, vocab_tgt_len):
    # Loading model using the 'build_transformer' function.
    # We will use the lengths of the source language and target language vocabularies, the 'seq_len', and the dimensionality of the embeddings
    model = build_transformer(vocab_src_len, vocab_tgt_len, config['seq_len'], config['seq_len'], config['d_model'], 
                              config['n_layer'], config['h_num'], config['dropout'], config['d_ff'])
    return model

def run_validation(model, valid_data, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in valid_data:
            src = batch.src.to(device)
            src_mask = batch.src_mask.to(device)
            label = batch.label.to(device)
            logits = model(src, src_mask)
            preds = torch.argmax(logits, dim=1)
            correct += (preds == label).sum().item()
            total += label.size(0)
    acc = correct / total if total > 0 else 0.0
    print(f'Validation accuracy: {acc:.4f}')
    return acc

class IdentityPositionalEncoding(nn.Module):
    """Identity positional encoding that does nothing."""
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x

if __name__ == "__main__":

    # get config
    config = get_config(DEBUG) # Retrieving config settings

    # Setting up device to run on GPU to train faster
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device {device}")

    # Data Preprocessing
    data = PrepareData(config['train_file'], config['dev_file'], config['batch_size'], UNK, PAD)
    sentiment_data = SentimentData('train.jsonl', config['batch_size'], data.en_word_dict, UNK, PAD)
    src_vocab_size = len(data.en_word_dict)
    tgt_vocab_size = len(data.cn_word_dict); print(f"tgt_vocab_size {tgt_vocab_size}")

    # Model
    pretrained_model = get_model(config, src_vocab_size, tgt_vocab_size).to(device)
    pretrained_model.load_state_dict(torch.load('./results/transformer-models/model-99.pt'))

    # pretrained_model.src_pos = IdentityPositionalEncoding().to(device)  # Use identity positional encoding
    # pretrained_model.tgt_pos = IdentityPositionalEncoding().to(device)  # Use identity positional encoding


    model = SentimentClassifier(pretrained_model, config['d_model'], num_classes=3).to(device)

    # Initializing CrossEntropyLoss function for training
    # We ignore padding tokens when computing loss, as they are not relevant for the learning process
    # We also apply label_smoothing to prevent overfitting
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], eps = 1e-9)

    # 记录每一轮的loss
    epoch_loss_list = []
    epoch_acc_list = []

    print(">>>>>>> start train")
    train_start = time.time()
    # Initializing epoch and global step variables
    initial_epoch = 0
    global_step = 0

    model_save_path = config['save_file']

    for epoch in range(initial_epoch, 100):
        # Initializing an iterator over the training dataloader
        # We also use tqdm to display a progress bar
        batch_iterator = tqdm(sentiment_data.train_data, desc = f'Processing epoch {epoch:02d}')
        epoch_loss = 0.0
        epoch_samples = 0

        for batch in batch_iterator:
            model.train()
            src = batch.src.to(device)
            src_mask = batch.src_mask.to(device)
            label = batch.label.to(device)
            
            logits = model(src, src_mask)

            loss = loss_fn(logits, label)

            # Clearing the gradients to prepare for the next batch
            optimizer.zero_grad()

            batch_size_actual = src.size(0)
            epoch_loss += loss.item() * batch_size_actual
            epoch_samples += batch_size_actual

            batch_iterator.set_postfix({"loss": f"{epoch_loss/epoch_samples:6.4f}"})

            loss.backward()
            
            # Updating parameters based on the gradients
            optimizer.step()
        
            global_step += 1

        avg_loss = epoch_loss / epoch_samples if epoch_samples > 0 else 0.0
        epoch_loss_list.append(avg_loss)

        print(f"Epoch {epoch} finished: loss = {avg_loss:.4f}")

        # ========== 验证集评估 ==========
        acc = run_validation(model, sentiment_data.valid_data, device)
        epoch_acc_list.append(acc)

        torch.save(model.state_dict(), f"{MODEL_DIR}/model-{epoch}.pt")

        # 每个epoch都追加保存loss和acc
        with open(f"{MODEL_DIR}/epoch_loss_list.txt", "a") as f_loss:
            f_loss.write(f"{avg_loss}\n")
        with open(f"{MODEL_DIR}/epoch_acc_list.txt", "a") as f_acc:
            f_acc.write(f"{acc}\n")

    # 保存loss和acc曲线数据到txt文件
    with open(f"{MODEL_DIR}/epoch_loss_list.txt", "w") as f_loss:
        for loss in epoch_loss_list:
            f_loss.write(f"{loss}\n")
    with open(f"{MODEL_DIR}/epoch_acc_list.txt", "w") as f_acc:
        for acc in epoch_acc_list:
            f_acc.write(f"{acc}\n")

    train_time = time.time() - train_start
    print(f"<<<<<<< finished train, cost {train_time:.4f} seconds")

    with open(f"{MODEL_DIR}/train_time.txt", "w") as f:
        f.write(f"{train_time}\n")
