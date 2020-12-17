
# coding: utf-8

# # Requirements

# In[22]:


get_ipython().system('pip install torch==1.5.0')
get_ipython().system('pip install numpy')


# # Imports

# In[23]:


import torch
import torch.nn as nn
import torch.optim as optim
import glob
import re
import random
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import numpy as np


# # Constants

# In[24]:


#is_cuda = torch.cuda.is_available()
is_cuda = False
MIN_TOKEN_FREQ = 5
PAD_LEN = 200
VAL_RATIO = 0.1
BATCH_SIZE = 128


# # Methods and classes

# In[25]:


def get_batch_times(data_size, batch_size=BATCH_SIZE, drop_last_batch=True):
    batch_times = data_size // BATCH_SIZE
    if not drop_last_batch:
        if data_size % BATCH_SIZE != 0 or data_size < BATCH_SIZE:
            batch_times += 1
    return batch_times


class Tokenizer(object):
    
    def __init__(self, token_idx_dic, pad_len):
        self.token_idx_dic = token_idx_dic
        self.pad_len = pad_len
        self.unk_idx = self.token_idx_dic['<unk>']
        self.pad_idx = self.token_idx_dic['<pad>']
        self.start_idx = self.token_idx_dic['<s>']
        self.end_idx = self.token_idx_dic['<e>']
            
    def __call__(self, text):
        l = len(text) + 2
        dl = l - self.pad_len
        if dl > 0:
            text = text[:self.pad_len-2]
        tokenized = [self.start_idx]
        tokenized.extend(list(map(lambda t: token_idx_dic.get(t, self.unk_idx), list(text))))
        tokenized.append(self.end_idx)
        if dl < 0:
            padding = [self.pad_idx] * abs(dl)
            tokenized.extend(padding)
        return tokenized

esp = 0.0000001

def batch_loss(model, data, batch_idx, batch_size=BATCH_SIZE):
    sidx = batch_idx*batch_size
    eidx = min(len(data), (batch_idx+1)*batch_size)
    batch = data[sidx:eidx]
    x_batch = torch.LongTensor(list(map(lambda item: item[0], batch)))
    y_batch = torch.LongTensor(list(map(lambda item: item[1], batch)))
    if is_cuda:
        x_batch = x_batch.cuda()
        y_batch = y_batch.cuda()
    y_pred = model(x_batch)
    loss = model.loss(y_pred, y_batch)
    return loss, y_pred

def measure_performance(preds, labels, cls_num):
    preds = np.array(preds)
    result_dic = dict()
    for cls_idx in range(cls_num):
        tp = tn = fp = fn = 0
        for p, l in zip(preds, labels):
            p = np.argmax(p)
            if cls_idx == l and p == cls_idx:
                tp += 1
            elif cls_idx == l and p != cls_idx:
                fn += 1
            elif cls_idx != l and p == cls_idx:
                fp += 1
            elif cls_idx != l and p != cls_idx:
                tn += 1
        acc = (tp + tn) / max(tp + tn + fp + fn, esp)
        prec = tp / max(tp + fp, esp)
        rec = tp / max(tp + fn, esp)
        f1 = (2*prec*rec) / max(prec+rec, esp)
        result_dic[cls_idx] = {
            'accuracy': acc,
            'precision': prec,
            'recall': rec,
            'f1score': f1
        }
    return result_dic

def save_model(model, path):
    torch.save(model.state_dict(), path)

def load_model(model, path):
    if is_cuda:
        model.load_state_dict(torch.load(path))
    else:
        model.load_state_dict(torch.load(path), map_location='cpu')

def binary_cross_entropy(pred, y):
    pred = nn.ReLU()(pred)
    loss = - (y * torch.log(pred + esp) + (1. - y) * torch.log(1. - pred + esp))
    return nn.ReLU()(loss)

def packed_to_rnn(rnn_cell, x, x_lengths):
    x = pack_padded_sequence(x, x_lengths.tolist(), batch_first=False, enforce_sorted=False)
    o, _ = rnn_cell(x)
    o = pad_packed_sequence(o, batch_first=False)[0]
    return o, _


# # Model classes

# In[26]:


class SingleLSTMModel(nn.Module):
    
    def __init__(self, max_token_idx, output_num, e_dim=32, h_dim=64, is_cuda=False):
        super().__init__()
        
        self.is_cuda = is_cuda
        self.output_num = output_num
        
        self.embedding = nn.Embedding(max_token_idx+1, e_dim)
        self.rnn = nn.LSTM(e_dim, h_dim, batch_first=False)
        self.out_linear = nn.Linear(h_dim, output_num)
        
    def __call__(self, x):
        x_len = torch.sum((x > 0).type(torch.IntTensor), -1)
        if self.is_cuda:
            x_len = x_len.cuda()
        x = x.transpose(0, 1)
        e = self.embedding(x)
        ho, (h, c) = packed_to_rnn(self.rnn, e, x_len)
        h = h[0]
        o = self.out_linear(h)
        y_pred = nn.Softmax(dim=-1)(o)
        return y_pred
    
    def loss(self, pred, target_cls_idx):
        onehot = torch.eye(self.output_num).repeat(target_cls_idx.shape[0], 1)[target_cls_idx]
        return torch.mean(binary_cross_entropy(pred, onehot))
    
    
class SingleBiLSTMModel(nn.Module):
    
    def __init__(self, max_token_idx, output_num, e_dim=32, h_dim=64, is_cuda=False):
        super().__init__()
        
        self.is_cuda = is_cuda
        self.output_num = output_num
        
        self.embedding = nn.Embedding(max_token_idx+1, e_dim)
        self.rnn = nn.LSTM(e_dim, h_dim, batch_first=False, bidirectional=True)
        self.out_linear = nn.Linear(h_dim, output_num)
        
    def __call__(self, x):
        x_len = torch.sum((x > 0).type(torch.IntTensor), -1)
        if self.is_cuda:
            x_len = x_len.cuda()
        x = x.transpose(0, 1)
        e = self.embedding(x)
        ho, (h, c) = packed_to_rnn(self.rnn, e, x_len)
        h = h[0]
        o = self.out_linear(h)
        y_pred = nn.Softmax(dim=-1)(o)
        return y_pred
    
    def loss(self, pred, target_cls_idx):
        onehot = torch.eye(self.output_num).repeat(target_cls_idx.shape[0], 1)[target_cls_idx]
        return torch.mean(binary_cross_entropy(pred, onehot))
    
    
class CNNBiLSTMModel(nn.Module):
    
    def __init__(self, max_token_idx, output_num, e_dim=32, h_dim=64, is_cuda=False):
        super().__init__()
        
        self.is_cuda = is_cuda
        self.output_num = output_num
        
        self.embedding = nn.Embedding(max_token_idx+1, e_dim)
        self.conv1d = nn.Conv1d(e_dim, h_dim, 3)
        self.rnn = nn.LSTM(h_dim, h_dim, batch_first=False, bidirectional=True)
        self.out_linear = nn.Linear(h_dim, output_num)
        
    def __call__(self, x):
        x_len = torch.sum((x > 0).type(torch.IntTensor), -1)
        if self.is_cuda:
            x_len = x_len.cuda()
        e = self.embedding(x).transpose(1, 2)
        c = self.conv1d(e).transpose(1, 2).transpose(0, 1)
        x_len = x_len - 2
        ho, (h, c) = packed_to_rnn(self.rnn, c, x_len)
        h = h[0]
        o = self.out_linear(h)
        y_pred = nn.Softmax(dim=-1)(o)
        return y_pred
    
    def loss(self, pred, target_cls_idx):
        onehot = torch.eye(self.output_num).repeat(target_cls_idx.shape[0], 1)[target_cls_idx]
        return torch.mean(binary_cross_entropy(pred, onehot))


# # Build class dictionary

# In[27]:


cls_idx_dic = dict()
idx_cls_dic = dict()
i = 0
dir_paths = glob.glob('data/*')
for i, dir_path in zip(range(len(dir_paths)), dir_paths):
    cls = dir_path.split('/')[1]
    cls_idx_dic[cls] = i
    idx_cls_dic[i] = cls
cls_num = i + 1
cls_idx_dic


# # Read input files

# In[28]:


data = []
err_cnt = 0
cls_cnt_dic = dict()
for dir_path in glob.glob('data/*'):
    cls = dir_path.split('/')[1]
    cls_idx = cls_idx_dic[cls]
    for text_path in glob.glob(dir_path + '/*'):
        with open(text_path, 'rb') as f:
            lines = f.readlines()
            try:
                lines = list(map(lambda item: item.decode('utf-8'), lines))    
            except UnicodeDecodeError as e1:
                try:
                    lines = list(map(lambda item: item.decode('euc-kr'), lines))
                except UnicodeDecodeError as e2:
                    err_cnt += 1
                    continue
            #정규표현식 기반의 데이터 가공
            lines = list(map(lambda line: re.sub('([A-Z]\s:)|([A-Z]\s)|([A-Z].\s)|<(.*)>|\((.*)\)|\.|\?|!|,|~|…|\n\n', '', line), lines))
            text = ''.join(lines)
            data.append((text, cls_idx))
            cnt = cls_cnt_dic.get(cls_idx, 0)
            cls_cnt_dic[cls_idx] = cnt + 1
data_size = len(data)
print(f'{data_size} files read.')
print(f'failed to read {err_cnt} files.')


# # Build tokenizer

# In[18]:


char_cnt_dic = dict()
for chs in list(map(lambda item: list(item[0]), data)):
    for ch in chs:
        cnt = char_cnt_dic.get(ch, 0)
        char_cnt_dic[ch] = cnt + 1
tokens = list(map(lambda item: item[0], 
                    list(filter(lambda kv: kv[1] > MIN_TOKEN_FREQ, char_cnt_dic.items()))))
token_idx_dic = dict()
token_idx_dic['<pad>'] = 0
token_idx_dic['<s>'] = 1
token_idx_dic['<e>'] = 2
max_token_idx = len(tokens)+3
for i, t in zip(range(3, max_token_idx), tokens):
    token_idx_dic[t] = i
token_idx_dic['<unk>'] = max_token_idx
tokenizer = Tokenizer(token_idx_dic, pad_len=PAD_LEN)


# # Split data

# In[19]:


random.shuffle(data)
train_bound = int(data_size * (1. - VAL_RATIO))
transformed_data = list(map(lambda kv: (tokenizer(kv[0]), kv[1]), data))
train_data = transformed_data[:train_bound]
val_data = transformed_data[train_bound:]
val_y = list(map(lambda item: item[1], val_data))
train_size = len(train_data)
val_size = len(val_data)


# # Oversampling

# In[20]:


def oversampling(train_data):
    oversampled_train_data = []
    max_cls_cnt = max(cls_cnt_dic.values())
    for cls_idx, cnt in cls_cnt_dic.items():
        short = max_cls_cnt - cnt
        target_data = list(filter(lambda item: item[1] == cls_idx, train_data))
        oversampled_train_data.extend(target_data)
        if short > 0:
            for i in range(short):
                oversampled_train_data.append(random.choice(target_data))
    random.shuffle(oversampled_train_data)
    print(f'{len(train_data)} -> {len(oversampled_train_data)}')
    return oversampled_train_data


# # Training and validation

# In[21]:


model = SingleLSTMModel(max_token_idx=max_token_idx, output_num=cls_num, is_cuda=is_cuda)
opt = optim.Adam(model.parameters())
for e in range(10):
    print(f'epoch {e+1} started...')
    model.train()
    oversampled_train_data = oversampling(train_data)
    for i in range(get_batch_times(len(oversampled_train_data), drop_last_batch=True)):
        loss, _ = batch_loss(model, oversampled_train_data, i)
        opt.zero_grad()
        loss.backward()
        opt.step()
        print(loss)
    val_loss_sum = 0.
    val_steps = 0
    model.eval()
    val_preds = []
    with torch.no_grad():
        for i in range(get_batch_times(len(val_data), drop_last_batch=False)):
            loss, preds = batch_loss(model, val_data, i)
            preds = preds.tolist()
            val_preds.extend(preds)
            val_loss_sum += loss * (len(preds) / BATCH_SIZE)
            val_steps += 1
    val_loss = val_loss_sum / val_steps
    val_performance = measure_performance(val_preds, val_y, cls_num=cls_num)
    print(e+1, val_loss)
    for k, v in val_performance.items():
        print(idx_cls_dic[k], v)
    save_model(model, f'{e+1}.model')
    print('===================================')

