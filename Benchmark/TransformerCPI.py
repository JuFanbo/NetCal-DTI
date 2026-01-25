
import math
device = 'cuda'
import torch.nn.functional as F, torch, torch.nn as nn,numpy as np,random
from dataloader import *
from torch_geometric.nn import GINConv
from torch.nn import Linear, ReLU, Sequential, Dropout, BatchNorm1d, MaxPool1d,GELU,Sigmoid, CrossEntropyLoss
from torch_geometric.data import Batch
def seq_emb_prot(seq_lst):
    seq_voc = "ABCDEFGHIKLMNOPQRSTUVWXYZ"
    seq_dict = {v: (i + 1) for i, v in enumerate(seq_voc)}
    lst = []
    for i in seq_lst:
        x = np.zeros(1200)
        for j, ch in enumerate(i[:1200]):
            x[j] = seq_dict[ch]
        lst.append(torch.LongTensor(x))
    return torch.stack(lst,dim=0).to('cuda')

def seq_emb_drug(seq_lst):
    seq_voc = "#%)(+-/.1032547698=A@CBEDGFIHKMLONPSRUTWVY[Z]\\acbedgfihmlonsruty"
    seq_dict = {v: (i + 1) for i, v in enumerate(seq_voc)}
    lst = []
    for i in seq_lst:
        x = np.zeros(64)
        for j, ch in enumerate(i[:64]):
            x[j] = seq_dict[ch]
        lst.append(torch.LongTensor(x))
    return torch.stack(lst,dim=0).to('cuda')
def pad(seq_lst,limit):
    lst = [F.pad(i,(0,0,0,limit-i.shape[0]),value=0) for i in seq_lst]
    return torch.stack(lst,dim=0)
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
        self.conv4 = GIN_block(dim,output_dim)

    def forward(self, mole):
        data = Batch.from_data_list(mole)
        x, edge_index, batch = data.x.to(device), data.edge_index.to(device), data.batch.to(device)
        x = self.conv1(x,edge_index)
        x = self.conv2(x,edge_index)
        x = self.conv3(x,edge_index)
        x = self.conv4(x,edge_index)
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

class SelfAttention(nn.Module):
    def __init__(self, hid_dim, n_heads, dropout, device):
        super().__init__()

        self.hid_dim = hid_dim
        self.n_heads = n_heads

        assert hid_dim % n_heads == 0

        self.w_q = nn.Linear(hid_dim, hid_dim)
        self.w_k = nn.Linear(hid_dim, hid_dim)
        self.w_v = nn.Linear(hid_dim, hid_dim)

        self.fc = nn.Linear(hid_dim, hid_dim)

        self.do = nn.Dropout(dropout)

        self.scale = torch.sqrt(torch.FloatTensor([hid_dim // n_heads])).to(device)

    def forward(self, query, key, value, mask=None):
        bsz = query.shape[0]

        # query = key = value [batch size, sent len, hid dim]

        Q = self.w_q(query)
        K = self.w_k(key)
        V = self.w_v(value)

        # Q, K, V = [batch size, sent len, hid dim]

        Q = Q.view(bsz, -1, self.n_heads, self.hid_dim // self.n_heads).permute(0, 2, 1, 3)
        K = K.view(bsz, -1, self.n_heads, self.hid_dim // self.n_heads).permute(0, 2, 1, 3)
        V = V.view(bsz, -1, self.n_heads, self.hid_dim // self.n_heads).permute(0, 2, 1, 3)

        # K, V = [batch size, n heads, sent len_K, hid dim // n heads]
        # Q = [batch size, n heads, sent len_q, hid dim // n heads]
        energy = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.scale

        # energy = [batch size, n heads, sent len_Q, sent len_K]
        if mask is not None:
            energy = energy.masked_fill(mask == 0, -1e10)

        attention = self.do(F.softmax(energy, dim=-1))

        # attention = [batch size, n heads, sent len_Q, sent len_K]

        x = torch.matmul(attention, V)

        # x = [batch size, n heads, sent len_Q, hid dim // n heads]

        x = x.permute(0, 2, 1, 3).contiguous()

        # x = [batch size, sent len_Q, n heads, hid dim // n heads]

        x = x.view(bsz, -1, self.n_heads * (self.hid_dim // self.n_heads))

        # x = [batch size, src sent len_Q, hid dim]

        x = self.fc(x)

        # x = [batch size, sent len_Q, hid dim]

        return x


class Encoder(nn.Module):
    """protein feature extraction."""
    def __init__(self, protein_dim=128, hid_dim=64, n_layers=3,kernel_size=5 , dropout=0.1, device='cuda'):
        super().__init__()

        assert kernel_size % 2 == 1, "Kernel size must be odd (for now)"

        self.input_dim = protein_dim
        self.hid_dim = hid_dim
        self.kernel_size = kernel_size
        self.dropout = dropout
        self.n_layers = n_layers
        self.device = device
        #self.pos_embedding = nn.Embedding(1000, hid_dim)
        self.scale = torch.sqrt(torch.FloatTensor([0.5])).to(device)
        self.convs = nn.ModuleList([nn.Conv1d(hid_dim, 2*hid_dim, kernel_size, padding=(kernel_size-1)//2) for _ in range(self.n_layers)])   # convolutional layers
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(self.input_dim,self.hid_dim)

    def forward(self, protein):
        #pos = torch.arange(0, protein.shape[1]).unsqueeze(0).repeat(protein.shape[0], 1).to(self.device)
        #protein = protein + self.pos_embedding(pos)
        #protein = [batch size, protein len,protein_dim]
        conv_input = self.fc(protein)
        # conv_input=[batch size,protein len,hid dim]
        #permute for convolutional layer
        conv_input = conv_input.permute(0, 2, 1)
        #conv_input = [batch size, hid dim, protein len]
        for i, conv in enumerate(self.convs):
            #pass through convolutional layer
            conved = conv(self.dropout(conv_input))
            #conved = [batch size, 2*hid dim, protein len]

            #pass through GLU activation function
            conved = F.glu(conved, dim=1)
            #conved = [batch size, hid dim, protein len]

            #apply residual connection / high way
            conved = (conved + conv_input) * self.scale
            #conved = [batch size, hid dim, protein len]

            #set conv_input to conved for next loop iteration
            conv_input = conved

        conved = conved.permute(0,2,1)
        # conved = [batch size,protein len,hid dim]
        return conved


class PositionwiseFeedforward(nn.Module):
    def __init__(self, hid_dim, pf_dim, dropout):
        super().__init__()

        self.hid_dim = hid_dim
        self.pf_dim = pf_dim

        self.fc_1 = nn.Conv1d(hid_dim, pf_dim, 1)  # convolution neural units
        self.fc_2 = nn.Conv1d(pf_dim, hid_dim, 1)  # convolution neural units

        self.do = nn.Dropout(dropout)

    def forward(self, x):
        # x = [batch size, sent len, hid dim]

        x = x.permute(0, 2, 1)

        # x = [batch size, hid dim, sent len]

        x = self.do(F.relu(self.fc_1(x)))

        # x = [batch size, pf dim, sent len]

        x = self.fc_2(x)

        # x = [batch size, hid dim, sent len]

        x = x.permute(0, 2, 1)

        # x = [batch size, sent len, hid dim]

        return x


class DecoderLayer(nn.Module):
    def __init__(self, hid_dim, n_heads, pf_dim, self_attention, positionwise_feedforward, dropout, device):
        super().__init__()

        self.ln = nn.LayerNorm(hid_dim)
        self.sa = self_attention(hid_dim, n_heads, dropout, device)
        self.ea = self_attention(hid_dim, n_heads, dropout, device)
        self.pf = positionwise_feedforward(hid_dim, pf_dim, dropout)
        self.do = nn.Dropout(dropout)

    def forward(self, trg, src, trg_mask=None, src_mask=None):
        # trg = [batch_size, compound len, atom_dim]
        # src = [batch_size, protein len, hid_dim] # encoder output
        # trg_mask = [batch size, compound sent len]
        # src_mask = [batch size, protein len]

        trg = self.ln(trg + self.do(self.sa(trg, trg, trg, trg_mask)))

        trg = self.ln(trg + self.do(self.ea(trg, src, src, src_mask)))

        trg = self.ln(trg + self.do(self.pf(trg)))

        return trg


class Decoder(nn.Module):
    """ compound feature extraction."""
    def __init__(self, atom_dim=128, hid_dim=64, n_layers=3, n_heads=8, pf_dim=256,
                 decoder_layer=DecoderLayer, self_attention=SelfAttention, positionwise_feedforward=PositionwiseFeedforward,
                 dropout=0.1, device='cuda'):
        super().__init__()
        self.ln = nn.LayerNorm(hid_dim)
        self.output_dim = atom_dim
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.pf_dim = pf_dim
        self.decoder_layer = decoder_layer
        self.self_attention = self_attention
        self.positionwise_feedforward = positionwise_feedforward
        self.dropout = dropout
        self.device = device
        self.sa = self_attention(hid_dim, n_heads, dropout, device)
        self.layers = nn.ModuleList(
            [decoder_layer(hid_dim, n_heads, pf_dim, self_attention, positionwise_feedforward, dropout, device)
             for _ in range(n_layers)])
        self.ft = nn.Linear(atom_dim, hid_dim)
        self.do = nn.Dropout(dropout)
        self.fc_1 = nn.Linear(hid_dim, 256)
        self.fc_2 = nn.Linear(256, 2)

    def forward(self, trg, src, trg_mask=None, src_mask=None):
        # trg = [batch_size, compound_len, atom_dim]
        # src = [batch_size, protein_len, hid_dim]  # encoder output

        trg = self.ft(trg)  # [batch_size, compound_len, hid_dim]

        for layer in self.layers:
            trg = layer(trg, src)  # [batch_size, compound_len, hid_dim]

        # Compute attention weights via L2 norm
        norm = torch.norm(trg, dim=2)  # [batch_size, compound_len]
        norm = F.softmax(norm, dim=1)  # [batch_size, compound_len]

        # Weighted sum over compound_len dimension (vectorized!)
        weighted_trg = trg * norm.unsqueeze(-1)  # [B, L, D] * [B, L, 1] → [B, L, D]
        sum = weighted_trg.sum(dim=1)  # [B, D]

        # trg = [batch size,hid_dim]
        label = F.relu(self.fc_1(sum))
        label = self.fc_2(label)
        return label


class TransformerCPI(nn.Module):
    def __init__(self,atom_dim=128):
        super().__init__()
        self.embedding_xt = nn.Embedding(26,128)
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.device = 'cuda'
        self.weight = nn.Parameter(torch.FloatTensor(atom_dim, atom_dim))
        self.init_weight()
        self.gin = GIN(atom_dim)

    def init_weight(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)


    def forward(self, emb1, emb2, seq, esm, smiles, mole):
        compound = self.gin(mole)
        protein = self.embedding_xt(seq_emb(seq))
        enc_src = self.encoder(protein)
        out = self.decoder(compound, enc_src)
        return out