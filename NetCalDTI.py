device = 'cuda'
import torch.nn.functional as F, torch, torch.nn as nn,numpy as np,random
from dataloader import *
from torch_geometric.nn import GINConv
from torch.nn import Linear, ReLU, Sequential, Dropout, BatchNorm1d, MaxPool1d,GELU,Sigmoid, CrossEntropyLoss
from torch_geometric.data import Batch
class ChannelAttention(nn.Module):
    def __init__(self, in_channels, ratio):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)

        self.fc = nn.Sequential(
            nn.Conv1d(in_channels, in_channels // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv1d(in_channels // ratio, in_channels, 1, bias=False)
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        out = self.sigmoid(out)
        return out * x
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv1d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.sigmoid(self.conv1(out))
        return out * x
class CBAM(nn.Module):
    def __init__(self, in_channels, ratio, kernel_size):
        super(CBAM, self).__init__()
        self.channelattention = ChannelAttention(in_channels, ratio=ratio)
        self.spatialattention = SpatialAttention(kernel_size=kernel_size)

    def forward(self, x):
        x = self.channelattention(x)
        x = self.spatialattention(x)
        return x
def pad(seq_lst,limit):
    lst = [F.pad(i,(0,0,0,limit-i.shape[0]),value=0) for i in seq_lst]
    return torch.stack(lst,dim=0)
def mlp(i,h,o,dropout=0.1):
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
            CBAM(o,ratio=4,kernel_size=3),
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


class NetCalDTI(torch.nn.Module):
    def __init__(self,k=128):
        super(NetCalDTI, self).__init__()
        self.conv = 40
        self.k = k
        self.attention_layer = nn.Linear(self.conv*4, self.conv*4)
        self.protein_attention_layer = nn.Linear(self.conv * 4, self.conv*4)
        self.drug_attention_layer = nn.Linear(self.conv*4,self.conv*4)
        self.cnn = CNN(conv=self.conv).to(device)
        self.gin = GIN(self.conv*4).to(device)
        self.relu = ReLU()
        self.sigmoid = Sigmoid()
        self.protein_fuser = mlp(self.conv*4+1152,1024,self.conv*4)
        self.dock = mlp(self.conv*8,1024,256)
        self.le_mlp = mlp(k*2,512,256,0.4)
        self.output = Sequential(
            mlp(512,256,128),
            Linear(128,2))


    def forward(self,emb1, emb2, seq, esm, smiles, mole):
        drugConv = self.gin(mole)
        proteinConv = self.cnn(seq)
        drug_att = self.drug_attention_layer(drugConv)
        protein_att = self.protein_attention_layer(proteinConv)
        drugConv = drugConv.permute(0, 2, 1)
        proteinConv = proteinConv.permute(0, 2, 1)
        d_att_layers = torch.unsqueeze(drug_att, 2).repeat(1, 1, proteinConv.shape[-1], 1)  # repeat along protein size
        p_att_layers = torch.unsqueeze(protein_att, 1).repeat(1, drugConv.shape[-1], 1, 1)  # repeat along drug size
        Atten_matrix = self.attention_layer(self.relu(d_att_layers + p_att_layers))
        Compound_atte = torch.mean(Atten_matrix, 2)
        Protein_atte = torch.mean(Atten_matrix, 1)
        Compound_atte = self.sigmoid(Compound_atte.permute(0, 2, 1))
        Protein_atte = self.sigmoid(Protein_atte.permute(0, 2, 1))
        drugConv = 0.5 * drugConv + drugConv * Compound_atte
        drugConv = torch.max(drugConv,dim=2,keepdim=False)[0]
        proteinConv = 0.5 * proteinConv + proteinConv * Protein_atte
        proteinConv = torch.max(proteinConv,dim=2,keepdim=False)[0]
        proteinConv = self.protein_fuser(torch.cat([proteinConv, esm], 1))
        x1 = self.le_mlp(torch.cat((emb1,emb2),1))
        x2 = self.dock(torch.cat([drugConv,proteinConv],1))
        x = torch.cat((x1,x2),1)
        return self.output(x)

