device = 'cuda'
import torch.nn.functional as F, torch, torch.nn as nn,numpy as np,random
from dataloader import *
from torch_geometric.nn import GINConv
from torch.nn import Linear, ReLU, Sequential, Dropout, BatchNorm1d, MaxPool1d,GELU,Sigmoid, CrossEntropyLoss
from torch_geometric.data import Batch

def pad(seq_lst,limit):
    lst = [F.pad(i,(0,0,0,limit-i.shape[0]),value=0) for i in seq_lst]
    return torch.stack(lst,dim=0)
def mlp(i,h,o,dropout=0.2):
    return Sequential(
        Linear(i, h),
        BatchNorm1d(h),
        GELU(),
        Dropout(dropout),

        Linear(h, o),
        BatchNorm1d(o),
        GELU(),
        Dropout(dropout)
    )
class GIN_block(nn.Module):
    def __init__(self, i,o):
        super(GIN_block, self).__init__()
        self.conv =  GINConv(Sequential(Linear(i, o),ReLU(),Linear(o, o)))
        self.linear = nn.Linear(i,o)
    def forward(self, x,edge_index):
        res = self.linear(x)
        x = self.conv(x,edge_index)
        return res + x
class GIN(nn.Module):
    def __init__(self,output_dim):
        super(GIN, self).__init__()
        num_features_xd = 79
        dim=int((output_dim+num_features_xd)*0.667)
        self.conv1 = GIN_block(num_features_xd,dim)
        self.conv2 = GIN_block(dim, dim)
        self.conv3 = GIN_block(dim, dim)
        self.conv4 = GIN_block(dim, dim)
        self.conv5 = GIN_block(dim,output_dim)

    def forward(self, mole):
        data = Batch.from_data_list(mole)
        x, edge_index, batch = data.x.to(device), data.edge_index.to(device), data.batch.to(device)
        x = self.conv1(x,edge_index)
        x = self.conv2(x,edge_index)
        x = self.conv3(x,edge_index)
        x = self.conv4(x,edge_index)
        x = self.conv5(x,edge_index)
        ptr = data.ptr
        x = [x[ptr[i]:ptr[i+1]] for i in range(len(ptr)-1)]
        x = pad(x,45)
        return x
def seq_emb(seq_lst):
    seq_voc = "ABCDEFGHIKLMNOPQRSTUVWXYZ"
    seq_dict = {v: (i + 1) for i, v in enumerate(seq_voc)}
    lst = []
    for i in seq_lst:
        x = np.zeros(1200)
        for j, ch in enumerate(i[:1200]):
            x[j] = seq_dict[ch]
        lst.append(torch.LongTensor(x))
    return torch.stack(lst,dim=0).to(device)
class cnn_block(nn.Module):
    def __init__(self, i, o):
        super(cnn_block, self).__init__()
        self.block = nn.Sequential(
            nn.Conv1d(i, o, kernel_size=3,padding=1),
            BatchNorm1d(o),
            ReLU(),
            nn.Conv1d(o, o, kernel_size=3, padding=1),
            BatchNorm1d(o),
            ReLU(),
            MaxPool1d(kernel_size=2, stride=2),
        )
        self.res = Sequential(
            nn.Conv1d(i, o, kernel_size=1,padding=0),
            MaxPool1d(kernel_size=2, stride=2))


    def forward(self, x):
        x = torch.add(self.block(x),self.res(x))
        return torch.relu(x)
class CNN(nn.Module):
    def __init__(self, conv=40):
        super(CNN, self).__init__()
        self.conv = conv
        self.embedding_xt = nn.Embedding(26,128)
        self.cnn0 = cnn_block(128,self.conv)
        self.cnn1 = cnn_block(self.conv,self.conv*2)
        self.cnn2 = cnn_block(self.conv*2,self.conv*4)
        self.cnn = Sequential(self.cnn0,self.cnn1,self.cnn2)
    def forward(self,seq_lst):
        seq = seq_emb(seq_lst)
        embedded_xt = self.embedding_xt(seq)
        embedded_xt = embedded_xt.permute(0,2,1)
        xt = self.cnn(embedded_xt)
        xt = xt.permute(0,2,1)
        return xt


class CNNGIN(torch.nn.Module):
    def __init__(self):
        super(CNNGIN, self).__init__()
        self.conv = 40
        self.cnn = CNN(conv=self.conv).to(device)
        self.gin = GIN(self.conv*4).to(device)
        self.protein_fuser = mlp(self.conv * 4 + 1152, 1024, self.conv * 4)
        self.dock = Sequential(
            mlp(self.conv * 8, 512, 256),
            Linear(256, 2)
        )

    def forward(self,emb1, emb2, seq, esm, smiles, mole):
        drugConv = self.gin(mole)
        proteinConv = self.cnn(seq)
        drugConv = torch.max(drugConv,dim=1,keepdim=False)[0]
        proteinConv = torch.max(proteinConv,dim=1,keepdim=False)[0]
        x = self.dock(torch.cat([drugConv,proteinConv],1))
        return  x

