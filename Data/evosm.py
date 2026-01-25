import pandas as pd
from esm.models.esmc import ESMC
from esm.sdk.api import *
import torch
import os
import pickle
from esm.tokenization import EsmSequenceTokenizer
from tqdm import tqdm
# 使用预下载的参数
os.environ["INFRA_PROVIDER"] = "True"
device = torch.device("cuda:0")
client = ESMC.from_pretrained("esmc_600m" ,device=device)

# 读取蛋白质序列，这里需要根据自己的数据格式进行调整
def read_seq(seqfilepath):
    with open(seqfilepath ,"r") as f:
        line = f.readline()
        seq = f.readline()
    return seq

# 这里沿用了上一次逆向出来的编码格式，可以替换为ESM自带的编码格式
all_amino_acid_number = {'A' :5, 'C' :23 ,'D' :13 ,'E' :9, 'F' :18,
                         'G' :6, 'H' :21 ,'I' :12 ,'K' :15 ,'L' :4,
                         'M' :20 ,'N' :17 ,'P' :14 ,'Q' :16 ,'R' :10,
                         'S' :8, 'T' :11 ,'V' :7, 'W' :22 ,'Y' :19,
                         '_' :32}
def esm_encoder_seq(seq, pad_len):
    s = [all_amino_acid_number[x] for x in seq]
    while len(s ) <pad_len:
        s.append(1)
    s.insert(0 ,0)
    s.append(2)
    return torch.tensor(s)

def get_esm_embedding(seq):
    protein_tensor = ESMProteinTensor(sequence=esm_encoder_seq(seq ,len(seq)).to(device))
    logits_output = client.logits(protein_tensor, LogitsConfig(sequence=True, return_embeddings=True))
    esm_embedding = logits_output.embeddings
    assert isinstance(esm_embedding ,torch.Tensor)
    return esm_embedding.squeeze().mean(dim=0).detach().cpu()

import pandas as pd
from esm.models.esmc import ESMC
from esm.sdk.api import *
import torch
import os
import pickle
from esm.tokenization import EsmSequenceTokenizer
from tqdm import tqdm
# 使用预下载的参数
os.environ["INFRA_PROVIDER"] = "True"
device = torch.device("cuda:0")
client = ESMC.from_pretrained("esmc_600m" ,device=device)

# 读取蛋白质序列，这里需要根据自己的数据格式进行调整
def read_seq(seqfilepath):
    with open(seqfilepath ,"r") as f:
        line = f.readline()
        seq = f.readline()
    return seq

# 这里沿用了上一次逆向出来的编码格式，可以替换为ESM自带的编码格式
all_amino_acid_number = {'A' :5, 'C' :23 ,'D' :13 ,'E' :9, 'F' :18,
                         'G' :6, 'H' :21 ,'I' :12 ,'K' :15 ,'L' :4,
                         'M' :20 ,'N' :17 ,'P' :14 ,'Q' :16 ,'R' :10,
                         'S' :8, 'T' :11 ,'V' :7, 'W' :22 ,'Y' :19,
                         '_' :32}
def esm_encoder_seq(seq, pad_len):
    s = [all_amino_acid_number[x] for x in seq]
    while len(s ) <pad_len:
        s.append(1)
    s.insert(0 ,0)
    s.append(2)
    return torch.tensor(s)

def get_esm_embedding(seq):
    protein_tensor = ESMProteinTensor(sequence=esm_encoder_seq(seq ,len(seq)).to(device))
    logits_output = client.logits(protein_tensor, LogitsConfig(sequence=True, return_embeddings=True))
    esm_embedding = logits_output.embeddings
    assert isinstance(esm_embedding ,torch.Tensor)
    return esm_embedding.squeeze().mean(dim=0).detach().cpu()

'''
to_add = ''
lst = pd.read_csv('seq.csv').values.tolist()
dic = {}
for seq in lst:
    dic[seq[0]] = seq[1]
emb = get_esm_embedding(dic[to_add])
with open('esm.pickle', 'rb') as f:
    a = pickle.load(f)
a[to_add] = emb
with open('esm.pickle', 'wb') as f:
    pickle.dump(a, f)
'''