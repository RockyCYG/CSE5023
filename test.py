from nltk import word_tokenize
import json

def load_data(train_file):
    ens = []
    labels = []
    max_len = 0
    with open(train_file, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line.strip())
            text = item['text'].strip().lower()  # 转小写并去除首尾空格
            label = int(item['label'])
            ens.append(["BOS"] + word_tokenize(text) + ["EOS"])
            labels.append(label)
            max_len = max(max_len, len(["BOS"] + word_tokenize(text) + ["EOS"]))
            x = len(["BOS"] + word_tokenize(text) + ["EOS"])
            if x >= 60:
                print(x, max_len)
    
    return ens, labels

load_data('train.jsonl')
