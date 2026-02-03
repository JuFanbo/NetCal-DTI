device = 'cuda'
from dataloader import *
from torch.nn import Linear,Sequential, Dropout, BatchNorm1d,GELU

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

class le(torch.nn.Module):
    def __init__(self,k=128):
        super(le, self).__init__()
        self.output = Sequential(
            mlp(k*2,512,128),
            Linear(128, 2)
        )

    def forward(self,emb1, emb2, seq, esm, smiles, mole):
        return self.output(torch.cat((emb1,emb2),1))

