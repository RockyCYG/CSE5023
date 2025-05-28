## CSE5023 Assignment2

## Instructions

Environment:

torch newset version (CUDA 12.6)

```bash
pip install numpy, matplotlib, tqdm, nltk, jieba
```

### 1.2 Training and BLEU Evaluation

```bash
python translator_en2cn.py
```

To test the final model on the test dataset, run `python model_test.py`.

### 1.3 Warm-up and Learning Rate Tuning

Uncomment the following two lines in `translator_en2cn.py`, then run `python translator_en2cn.py` to enable warm-up and learning rate tuning.

```python
# scheduler = CosineLRScheduler(optimizer, t_initial=100, lr_min=1e-6, warmup_t=3, warmup_lr_init=1e-7, t_in_epochs=True, initialize=True)
# scheduler.step(epoch)
```

### 1.4 Ablation Study of Hyper-parameters

Change the configuration in `translator_en2cn.py`, then run `python translator_en2cn.py` to enable warm-up and learning rate tuning.

```json
{
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
```

### 1.5 Positional Embedding

Comment out the following two lines in `transformer.py`:
```python
src_pos = PositionalEncoding(d_model, src_seq_len, dropout)
tgt_pos = PositionalEncoding(d_model, tgt_seq_len, dropout)
```
and uncomment the next two lines:
```python
# src_pos = LearnablePositionalEncoding(d_model, src_seq_len, dropout)
# tgt_pos = LearnablePositionalEncoding(d_model, tgt_seq_len, dropout)
```
Then run `python translator_en2cn.py`.

### 2.1 Fine-tuning

``` bash
python fine_tuning.py
```

To test the final model, run `python sentiment_model_test.py`.

### 2.2 Positional Embedding

Uncomment the following two lines in `fine_tuning.py`:
```python
# pretrained_model.src_pos = IdentityPositionalEncoding().to(device)
# pretrained_model.tgt_pos = IdentityPositionalEncoding().to(device)
```
Then run `python fine_tuning.py`.

### 3.1 ViT Implementation

``` bash
python vit_train.py
```

To test the final model, run `python eval_vit.py`.

### 3.2 Comparison with CNNs

``` bash
python cnn.py
```

To test the final model, run `python eval_cnn.py`.

All the results above will be saved at **results** folder. There are also many files provided for plotting loss and accuracy curves.
