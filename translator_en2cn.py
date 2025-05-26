#!/usr/bin/env python
# coding: utf-8

# Data Preparation: English-to-Chinese Translator Data

from model.transformer import build_transformer
from tokenization import PrepareData, MaskBatch
import time
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import os

import nltk
from nltk.translate.bleu_score import corpus_bleu, sentence_bleu, SmoothingFunction
import jieba

# nltk.download('punkt_tab')

import warnings
warnings.filterwarnings('ignore') # Filtering warnings

# from timm.scheduler.cosine_lr import CosineLRScheduler


# init parameters
PAD = 0  # padding word-id
UNK = 1  # unknown word-id


# DEBUG = True    # Debug / Learning Purposes.
DEBUG = False # Build the model, better with GPU CUDA enabled.

MODEL_DIR = "results/transformer-models-learnable"
os.makedirs(MODEL_DIR, exist_ok=True)

def get_config(debug=True):
    if debug:
        return{
            'lr': 1e-2,
            'batch_size': 64,
            'num_epochs': 2,
            'n_layer': 3,
            'h_num': 8,
            'd_model': 128, # Dimensions of the embeddings in the Transformer
            'd_ff': 256, # Dimensions of the feedforward layer in the Transformer
            'dropout': 0.1,
            'seq_len': 120, # max length
            'train_file': 'data/en-cn/train_mini.txt',
            'dev_file': 'data/en-cn/dev_mini.txt',
            'save_file': f'{MODEL_DIR}/model.pt'
        }
    else:
        return{
            'lr': 1e-4,
            'batch_size': 64,
            'num_epochs': 20,
            'n_layer': 6,
            'h_num': 8,
            'd_model': 256, # Dimensions of the embeddings in the Transformer
            'd_ff': 1024, # Dimensions of the feedforward layer in the Transformer
            'dropout': 0.1,
            'seq_len': 120, # max length
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
model = get_model(config, src_vocab_size, tgt_vocab_size).to(device)



# Initializing CrossEntropyLoss function for training
# We ignore padding tokens when computing loss, as they are not relevant for the learning process
# We also apply label_smoothing to prevent overfitting
loss_fn = nn.CrossEntropyLoss(ignore_index=PAD, label_smoothing=0.).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], eps = 1e-9)
# scheduler = CosineLRScheduler(optimizer, t_initial=100, lr_min=1e-6, warmup_t=3, warmup_lr_init=1e-7, t_in_epochs=True, initialize=True)


def casual_mask(size):
    # Creating a square matrix of dimensions 'size x size' filled with ones
    mask = torch.triu(torch.ones(1, size, size), diagonal = 1).type(torch.int)
    return mask == 0


# Define function to obtain the most probable next token
def greedy_decode(model, source, source_mask, tokenizer_tgt, max_len, device):
    # Retrieving the indices from the start and end of sequences of the target tokens
    bos_id = tokenizer_tgt.get('BOS')
    eos_id = tokenizer_tgt.get('EOS')
    
    # Computing the output of the encoder for the source sequence
    encoder_output = model.encode(source, source_mask)
    # Initializing the decoder input with the Start of Sentence token
    decoder_input = torch.empty(1,1).fill_(bos_id).type_as(source).to(device)
    
    # Looping until the 'max_len', maximum length, is reached
    while True:
        if decoder_input.size(1) == max_len:
            break
            
        # Building a mask for the decoder input
        decoder_mask = casual_mask(decoder_input.size(1)).type_as(source_mask).to(device)
        
        # Calculating the output of the decoder
        out = model.decode(encoder_output, source_mask, decoder_input, decoder_mask)
        
        # Applying the projection layer to get the probabilities for the next token
        prob = model.project(out[:, -1])
        
        # Selecting token with the highest probability
        _, next_word = torch.max(prob, dim=1)
        decoder_input = torch.cat([decoder_input, torch.empty(1,1). type_as(source).fill_(next_word.item()).to(device)], dim=1)
        
        # If the next token is an End of Sentence token, we finish the loop
        if next_word == eos_id:
            break
            
    return decoder_input.squeeze(0) # Sequence of tokens generated by the decoder



def cut_word(sentence):
    words = list(jieba.cut(sentence, cut_all=False))
    return words


# 记录每一轮的loss和bleu
epoch_loss_list = []
epoch_bleu_list = []

# Defining function to evaluate the model on the validation dataset
# num_examples = 2, two examples per run
def run_validation(model, data, tokenizer_tgt, max_len, device, print_msg, num_examples=4):
    model.eval()
    count = 0
    console_width = 80
    results = []

    with torch.no_grad():
        for i, batch in enumerate(data.dev_data):
            count += 1
            encoder_input = batch.src.to(device)
            encoder_mask = batch.src_mask.to(device)
            assert encoder_input.size(0) ==  1, 'Batch size must be 1 for validation.'
            model_out = greedy_decode(model, encoder_input, encoder_mask, tokenizer_tgt, max_len, device)

            source_text = " ".join([data.en_index_dict[w] for w in data.dev_en[i]])
            target_text = " ".join([data.cn_index_dict[w] for w in data.dev_cn[i]])

            model_out_text = []
            for j in range(1, model_out.size(0)):
                sym = data.cn_index_dict[model_out[j].item()]
                if sym != 'EOS':
                    model_out_text.append(sym)
                else:
                    break

            prediction = cut_word("".join(model_out_text))
            reference = cut_word("".join([data.cn_index_dict[w] for w in data.dev_cn[i][1:-1]]))
            # 如果预测或参考中有UNK，跳过本条
            if 'UNK' in source_text or 'UNK' in target_text or 'UNK' in prediction or 'UNK' in reference:
                continue
            predictions = prediction
            references = [reference]
            # print_msg('-'*console_width)
            # print_msg(f'SOURCE: {source_text}')
            # print_msg(f'TARGET: {target_text}')
            # print_msg(f'PREDICTED: {model_out_text}')
            # print_msg(f'PREDICTION: {predictions}')
            # print_msg(f'REFERENCE: {references}')
            if len(predictions) == 1:
                weights = (1.0,)
            elif len(predictions) == 2:
                weights = (0.5, 0.5)
            elif len(predictions) == 3:
                weights = (1 / 3, 1 / 3, 1 / 3)
            else:
                weights = (0.25, 0.25, 0.25, 0.25)
            bleu = sentence_bleu(references, predictions, weights=weights, smoothing_function=SmoothingFunction().method4)
            # print_msg(f"BLEU score: {bleu}")
            results.append(bleu)

    avg_bleu = np.average(results) if results else 0.0
    print_msg(f'Average BLEU score on validation set: {avg_bleu:.4f}')
    return avg_bleu


# Training model
print(">>>>>>> start train")
train_start = time.time()

# Initializing epoch and global step variables
initial_epoch = 0
global_step = 0

model_save_path = config['save_file']

# Iterating over each epoch from the 'initial_epoch' variable up to the number of epochs informed in the config
for epoch in range(initial_epoch, 100):
    # Initializing an iterator over the training dataloader
    # We also use tqdm to display a progress bar
    batch_iterator = tqdm(data.train_data, desc = f'Processing epoch {epoch:02d}')
    epoch_loss = 0.0 # Initializing epoch loss
    epoch_samples = 0 # Initializing epoch samples
    
    # For each batch...
    for batch in batch_iterator:
        model.train() # Train the model
        
        # Loading input data and masks onto the GPU
        encoder_input = batch.src.to(device)
        decoder_input = batch.tgt.to(device)
        encoder_mask = batch.src_mask.to(device)
        decoder_mask = batch.tgt_mask.to(device)
        # print(encoder_input[0], encoder_mask[0], decoder_input[0], decoder_mask[0])
        # print(encoder_input.shape, encoder_mask.shape, decoder_input.shape, decoder_mask.shape)

        optimizer.zero_grad()

        # Running tensors through the Transformer
        encoder_output = model.encode(encoder_input, encoder_mask)
        decoder_output = model.decode(encoder_output, encoder_mask, decoder_input, decoder_mask)
        proj_output = model.project(decoder_output)
        
        label = batch.tgt_y.to(device)
        loss = loss_fn(proj_output.view(-1, tgt_vocab_size), label.view(-1))
        loss.backward()
        optimizer.step()
        batch_size_actual = encoder_input.size(0)
        epoch_loss += loss.item() * batch_size_actual
        epoch_samples += batch_size_actual
        batch_iterator.set_postfix({"loss": f"{epoch_loss/epoch_samples:6.4f}"})
        global_step += 1 # Updating global step count

    # Updating the learning rate
    # scheduler.step(epoch)

    avg_loss = epoch_loss / epoch_samples if epoch_samples > 0 else 0.0
    epoch_loss_list.append(avg_loss)

    if (epoch + 1) % 10 == 0:
        avg_bleu = run_validation(model, data, data.cn_word_dict, config['seq_len'], device, lambda msg: batch_iterator.write(msg))
        epoch_bleu_list.append(avg_bleu)
        print(f"Epoch {epoch} finished: loss = {avg_loss:.4f}, bleu = {avg_bleu:.4f}")
    else:
        print(f"Epoch {epoch} finished: loss = {avg_loss:.4f}")
        avg_bleu = None  # 保证每一行都有对应bleu

    torch.save(model.state_dict(), f"{MODEL_DIR}/model-{epoch}.pt")

    # 每个epoch都追加保存loss和bleu
    with open(f"{MODEL_DIR}/epoch_loss_list.txt", "a") as f_loss:
        f_loss.write(f"{avg_loss}\n")
    with open(f"{MODEL_DIR}/epoch_bleu_list.txt", "a") as f_bleu:
        if avg_bleu is not None:
            f_bleu.write(f"{avg_bleu}\n")
        else:
            f_bleu.write("\n")

# 保存loss和bleu曲线数据到txt文件
with open(f"{MODEL_DIR}/epoch_loss_list.txt", "w") as f_loss:
    for loss in epoch_loss_list:
        f_loss.write(f"{loss}\n")
with open(f"{MODEL_DIR}/epoch_bleu_list.txt", "w") as f_bleu:
    for bleu in epoch_bleu_list:
        f_bleu.write(f"{bleu}\n")

train_time = time.time() - train_start
print(f"<<<<<<< finished train, cost {train_time:.4f} seconds")

with open(f"{MODEL_DIR}/train_time.txt", "w") as f:
    f.write(f"{train_time}\n")
