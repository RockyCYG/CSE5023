import torch
import json
import numpy as np
from tqdm import tqdm
from nltk import word_tokenize
from model.transformer import build_transformer
from tokenization import PrepareData, seq_padding
from fine_tuning import SentimentClassifier, get_config, get_model

def load_data(file):
    ens = []
    labels = []
    with open(file, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line.strip())
            text = item['text'].strip().lower()  # 转小写并去除首尾空格
            label = int(item['label'])
            ens.append(["BOS"] + word_tokenize(text) + ["EOS"])
            labels.append(label)
    
    return ens, labels

# init parameters
PAD = 0  # padding word-id
UNK = 1  # unknown word-id

# DEBUG = True    # Debug / Learning Purposes.
DEBUG = False  # Build the model, better with GPU CUDA enabled.

# get config
config = get_config(DEBUG) # Retrieving config settings

# Setting up device to run on GPU to train faster
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device {device}")

# Data Preprocessing
data = PrepareData(config['train_file'], config['dev_file'], config['batch_size'], UNK, PAD)
src_vocab_size = len(data.en_word_dict); print(f"src_vocab_size {src_vocab_size}")
tgt_vocab_size = len(data.cn_word_dict); print(f"tgt_vocab_size {tgt_vocab_size}")

# Model
pretrained_model = get_model(config, src_vocab_size, tgt_vocab_size).to(device)
pretrained_model.load_state_dict(torch.load('./resultstransformer-models/model-99.pt'))

model = SentimentClassifier(pretrained_model, config['d_model'], num_classes=3).to(device)
model.load_state_dict(torch.load('./resultssentiment-models/model-49.pt'))

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
    def __init__(self, train_file, batch_size, en_dict, unk_id, pad_id):
        self.unk_id = unk_id
        self.pad_id = pad_id
        self.train_en, self.train_label = self.load_data(train_file)
        self.train_en, self.train_label = self.wordToID(self.train_en, self.train_label, en_dict)
        self.train_data = self.splitBatch(self.train_en, self.train_label, batch_size, shuffle=False)


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
        return out_en_ids, out_labels
        
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

# 定义标签映射
label_mapping = {
    0: "negative",
    1: "neutral",
    2: "positive"
}

# 设置模型为评估模式
model.eval()

# 准备测试数据
test_data = SentimentData(
    train_file='./test.jsonl',  # 实际应为测试文件
    batch_size=config['batch_size'],
    en_dict=data.en_word_dict,
    unk_id=UNK,
    pad_id=PAD
)

def test_model(model, test_data, device):
    """按批次测试模型并输出每条样本的详细结果，同时计算不含未知词样本的准确率"""
    model.eval()
    all_predictions = []
    all_labels = []
    valid_predictions = []  # 不含未知词的预测结果
    valid_labels = []       # 不含未知词的真实标签
    
    print("\n===== 情感分类预测结果 =====")
    
    with torch.no_grad():
        for i, batch in enumerate(tqdm(test_data.train_data, desc="Testing")):
            # 准备输入
            src = batch.src.to(device)
            src_mask = batch.src_mask.to(device)
            true_labels = batch.label.to(device)
            
            # 模型推理
            logits = model(src, src_mask)
            predictions = torch.argmax(logits, dim=1)
            
            # 收集所有结果
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(true_labels.cpu().numpy())
            
            # 输出该批次所有样本的详细结果
            for j in range(len(predictions)):
                sample_idx = i * config['batch_size'] + j
                text_tokens = test_data.train_en[sample_idx]
                
                # 检查是否包含未知词（ID=1）
                has_unknown = 1 in text_tokens
                
                # 将ID转换为单词（跳过BOS/EOS和PAD）
                text = ' '.join([data.en_index_dict.get(token, f'UNK_{token}') 
                                for token in text_tokens 
                                if token not in [data.en_word_dict['BOS'], 
                                                data.en_word_dict['EOS'], 
                                                PAD]])
                
                # 记录有效样本（不含未知词）
                if not has_unknown:
                    valid_predictions.append(predictions[j].item())
                    valid_labels.append(true_labels[j].item())
                
                    print(f"\n样本 {sample_idx+1} ({'含未知词' if has_unknown else '有效'}):")
                    print(f"文本: {text}")
                    print(f"预测标签: {label_mapping[predictions[j].item()]} ({predictions[j].item()})")
                    print(f"真实标签: {label_mapping[true_labels[j].item()]} ({true_labels[j].item()})")
                    print(f"预测是否正确: {'✅' if predictions[j].item() == true_labels[j].item() else '❌'}")
                    print("-" * 60)
    
    # 计算总体准确率
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    all_accuracy = np.mean(all_predictions == all_labels)
    
    # 计算有效样本（不含未知词）的准确率
    valid_predictions = np.array(valid_predictions)
    valid_labels = np.array(valid_labels)
    valid_accuracy = np.mean(valid_predictions == valid_labels) if len(valid_labels) > 0 else 0
    
    # 输出各类别的准确率
    print("\n===== 总体准确率统计 =====")
    print(f"全部样本准确率: {all_accuracy:.4f} ({np.sum(all_predictions == all_labels)}/{len(all_labels)})")
    print(f"有效样本（不含未知词）准确率: {valid_accuracy:.4f} ({np.sum(valid_predictions == valid_labels)}/{len(valid_labels)})")
    
    # 输出有效样本中各类别的准确率
    if len(valid_labels) > 0:
        print("\n===== 有效样本中各类别的准确率 =====")
        for label_id, label_name in label_mapping.items():
            mask = (valid_labels == label_id)
            if np.sum(mask) > 0:
                class_accuracy = np.mean(valid_predictions[mask] == valid_labels[mask])
                print(f"{label_name} ({label_id}): {class_accuracy:.4f} ({np.sum(valid_predictions[mask] == valid_labels[mask])}/{np.sum(mask)})")
    
    return all_accuracy, valid_accuracy

# 运行测试
accuracy = test_model(model, test_data, device)
