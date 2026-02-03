import scipy.sparse as sp,scipy.sparse.linalg as li,\
    random,networkx as nx,pandas as pd,pickle,torch,numpy as np,os
from Data.mole_pyg import pyG_data
device = 'cuda'
class Loader:
    def __init__(self,dti_file,redo_le=True,save_test=True,k=128):
        ds = 'drugbank'
        if 'davis' in dti_file:
            ds = 'davis'
        if 'bindingdb' in dti_file:
            ds = 'bindingdb'
        mole_file = f'Data/{ds}.pickle'
        with open(mole_file ,'rb') as f:
            self.mole_dic = pickle.load(f)
        lst = pd.read_csv('Data/seq.csv').values.tolist()
        self.seq_dic = {}
        for i in lst:
            self.seq_dic[i[0]] = i[1]
        lst = pd.read_csv('Data/smiles.csv').values.tolist()
        self.smiles_dic = {}
        for i in lst:
            self.smiles_dic[i[0]] = i[1]
        with open('Data/esm.pickle','rb') as f:
            self.esm_dic = pickle.load(f)
        self.DTIs = pd.read_csv(dti_file).values.tolist()
        self.k = k
        self.nodes = list(set([i[0] for i in self.DTIs]))+list(set([i[1] for i in self.DTIs]))
        random.shuffle(self.DTIs)
        x = int(0.1 * len(self.DTIs))
        self.train = self.DTIs[:8 * x]
        self.vali = self.DTIs[8 * x:9 * x]
        self.test = self.DTIs[9 * x:]
        df = pd.DataFrame(self.test)
        if save_test:
            df.to_csv('Data/test.csv',index=False)
        self.backup_mole = pyG_data('CC')
        if redo_le:
            _G = nx.Graph()
            for i in self.nodes:
                _G.add_edge(i, i)
            for i in random.sample(self.train,int(0.5*len(self.train))):
                if i[2] == 1:
                    _G.add_edge(i[0], i[1])
            self.network_embedding = self.le(_G)
            with open('le.pickle','wb') as f:
                pickle.dump(self.network_embedding,f)
        else:
            with open('le.pickle','rb') as f:
                self.network_embedding = pickle.load(f)


    def prepare_data(self,lst):
        emb1,emb2,seq,esm,smiles,mole,y = [],[],[],[],[],[],[]
        for i in lst:
            y.append(float(i[2]))
            emb1.append(self.network_embedding[i[0]] if i[0] in self.network_embedding else torch.zeros(128))
            emb2.append(self.network_embedding[i[1]] if i[1] in self.network_embedding else torch.zeros(128))
            seq.append(self.seq_dic[i[0]] if i[0] in self.seq_dic else 'AAA')
            smiles.append(self.smiles_dic[i[1]] if i[1] in self.smiles_dic else 'AAA')
            if i[0] in self.esm_dic:
                esm.append(self.esm_dic[i[0]])
            else:
                esm.append(torch.zeros(1152))
            if i[1] in self.mole_dic:
                mole.append(self.mole_dic[i[1]])
            else:
                mole.append(self.backup_mole)
        emb1 = torch.stack(emb1, dim=0).float().to(device)
        emb2 = torch.stack(emb2, dim=0).float().to(device)
        esm = torch.stack(esm, dim=0).float().to(device)
        y = torch.tensor(y).long().to(device)
        return emb1,emb2,seq,esm,smiles,mole,y

    def le(self,G):
        nodes = list(G.nodes())
        i2n = {}
        n2i = {}
        for i in range(len(nodes)):
            i2n[i] = nodes[i]
            n2i[nodes[i]] = i
        G = nx.relabel_nodes(G, n2i)
        emb_dic = {}
        A = nx.adjacency_matrix(G).astype(float)
        degrees = dict(G.degree())
        D_inv_sqrt = sp.diags(1 / np.sqrt(list(degrees.values())).clip(1), dtype=float)
        L = sp.eye(G.number_of_nodes()) - D_inv_sqrt @ A @ D_inv_sqrt
        L = sp.csr_matrix(L)
        EigVal, EigVec = li.eigsh(L, return_eigenvectors=True, k=self.k, which="SA")
        _X = np.real(EigVec)
        for i in i2n:
            emb_dic[i2n[i]] = torch.tensor(_X[i].tolist()).float().cpu()
        return emb_dic

