import scipy.sparse as sp,scipy.sparse.linalg as li,\
    random,networkx as nx,pandas as pd,pickle,torch,numpy as np,os
from Data.mole_pyg import pyG_data
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator
from tqdm import tqdm
import numpy as np,torch
device = 'cuda'


def find_similar_drugs(query_smiles, smiles_dict, k):
    query_mol = Chem.MolFromSmiles(query_smiles)
    fp_gen = rdFingerprintGenerator.GetMorganGenerator(radius=2,fpSize=64)
    query_fp = fp_gen.GetFingerprint(query_mol)
    results = []
    for drug_id, smiles in smiles_dict.items():
        try:
            mol = Chem.MolFromSmiles(smiles)
            fp = fp_gen.GetFingerprint(mol)
            sim = DataStructs.TanimotoSimilarity(query_fp, fp)
            results.append((drug_id, sim))
        except:
            pass
    results.sort(key=lambda x: x[1], reverse=True)
    return [drug_id for drug_id, _ in results[:k]]


def find_similar_proteins(query_vector, protein_vectors, k):
    query_vector = query_vector.flatten()
    sim_scores = {}
    for pid, vec in protein_vectors.items():
        vec = vec.flatten()
        sim = torch.nn.functional.cosine_similarity(query_vector, vec, dim=0)
        sim_scores[pid] = sim.item()
    return sorted(sim_scores, key=sim_scores.get, reverse=True)[:k]

def split(dti_lst,target_cold, drug_cold):
    random.shuffle(dti_lst)
    limit = len(dti_lst)//10
    collected,vali,test = [],[],[]
    for i in range(len(dti_lst)):
        if i in collected:
            continue
        if len(vali) > limit:
            break
        vali.append(dti_lst[i])
        collected.append(i)
        if target_cold:
            for j in range(len(dti_lst)):
                if dti_lst[j][0] == dti_lst[i][0]:
                    vali.append(dti_lst[j])
                    collected.append(j)
        if drug_cold:
            for j in range(len(dti_lst)):
                if dti_lst[j][1] == dti_lst[i][1]:
                    vali.append(dti_lst[j])
                    collected.append(j)
    for i in range(len(dti_lst)):
        if i in collected:
            continue
        if len(test) > limit:
            break
        test.append(dti_lst[i])
        collected.append(i)
        if target_cold:
            for j in range(len(dti_lst)):
                if dti_lst[j][0] == dti_lst[i][0]:
                    test.append(dti_lst[j])
                    collected.append(j)
        if drug_cold:
            for j in range(len(dti_lst)):
                if dti_lst[j][1] == dti_lst[i][1]:
                    test.append(dti_lst[j])
                    collected.append(j)
    train = [dti_lst[i] for i in range(len(dti_lst)) if i not in collected]
    return train,vali,test


class Loader:
    def __init__(self,dti_file,redo_le=True,save_test=True,target_cold=False,drug_cold=False,lap_dim=128,k=2):
        ds = 'drugbank'
        if 'davis' in dti_file:
            ds = 'davis'
        if 'bindingdb' in dti_file:
            ds = 'bindingdb'
        mole_file = f'Data/{ds}.pickle'
        with open(mole_file ,'rb') as f:
            self.mole_dic = pickle.load(f)
        with open('Data/esm.pickle','rb') as f:
            self.esm_dic = pickle.load(f)
        self.lap_dim = lap_dim
        self.k = k
        self.drug_sim_df = pd.read_csv('./Data/drug_top10_similarity.csv')
        self.protein_sim_df = pd.read_csv('./Data/protein_top10_similarity.csv')
        dti_lst = pd.read_csv(dti_file).values.tolist()
        self.targets = list(set([i[0] for i in dti_lst]))
        self.drugs = list(set([i[1] for i in dti_lst]))
        lst = pd.read_csv('Data/seq.csv').values.tolist()
        self.seq_dic = {}
        for i in lst:
            self.seq_dic[i[0]] = i[1]
        lst = pd.read_csv('Data/smiles.csv').values.tolist()
        self.smiles_dic = {}
        for i in lst:
            self.smiles_dic[i[0]] = i[1]
        self.train,self.vali,self.test = split(dti_lst,target_cold, drug_cold)
        df = pd.DataFrame(self.test)
        if save_test:
            df.to_csv('Data/test.csv',index=False)
        self.backup_mole = pyG_data('CC')
        if redo_le:
            _G = nx.Graph()
            for i in list(set([j[0] for j in self.train]+[j[1] for j in self.train])):
                _G.add_edge(i,i)
            for i in random.sample(self.train,int(0.5*len(self.train))):
                if i[2] == 1:
                    _G.add_edge(i[0], i[1])
            self.network_embedding = self.le(_G)
            self.computed_keys = list(self.network_embedding.keys())
            print("checking targets...")
            for i in tqdm(self.targets):
                if not i in self.computed_keys:
                    self.network_embedding[i] = self.target_nearest_k(i,k)
            print("checking drugs...")
            for i in tqdm(self.drugs):
                if not i in self.computed_keys:
                    self.network_embedding[i] = self.drug_nearest_k(i,k)
            with open('le.pickle','wb') as f:
                pickle.dump(self.network_embedding,f)
        else:
            with open('le.pickle','rb') as f:
                self.network_embedding = pickle.load(f)

    def target_nearest_k(self,target_id,k):
        try:
            matches = self.protein_sim_df[self.protein_sim_df['query_protein'] == target_id]
            k_neighbors = matches[matches['similar_protein'].isin(self.computed_keys)]
            k_neighbors = k_neighbors.head(k)[['similar_protein', 'similarity']].values.tolist()
            tensors = [self.network_embedding[i[0]] for i in k_neighbors]
        except:
            try:
                dict = {i:self.esm_dic[i] for i in self.esm_dic if i in self.computed_keys}
                tensors = find_similar_proteins(self.esm_dic[target_id], dict,self.k)
            except:
                return torch.zeros(self.lap_dim)
        return torch.stack(tensors).mean(dim=0) if len(tensors)>0 else torch.zeros(self.lap_dim)

    def drug_nearest_k(self, drug_id, k):
        try:
            matches = self.drug_sim_df[self.drug_sim_df['query_drug'] == drug_id]
            k_neighbors = matches[matches['similar_drug'].isin(self.computed_keys)]
            k_neighbors = k_neighbors.head(k)[['similar_drug', 'similarity']].values.tolist()
            tensors = [self.network_embedding[i[0]] for i in k_neighbors]
        except:
            try:
                dict = {i: self.smiles_dic[i] for i in self.smiles_dic if i in self.computed_keys}
                tensors = find_similar_drugs(self.smiles_dic[drug_id], dict,self.k)
            except:
                return torch.zeros(self.lap_dim)
        return torch.stack(tensors).mean(dim=0) if len(tensors)>0 else torch.zeros(self.lap_dim)


    def prepare_data(self,lst):
        emb1,emb2,seq,esm,smiles,mole,y = [],[],[],[],[],[],[]
        for i in lst:
            y.append(float(i[2]))
            emb1.append(self.network_embedding[i[0]] if i[0] in self.network_embedding else self.target_nearest_k(i[0],self.k))
            emb2.append(self.network_embedding[i[1]] if i[1] in self.network_embedding else self.drug_nearest_k(i[1],self.k))
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
        EigVal, EigVec = li.eigsh(L, return_eigenvectors=True, k=self.lap_dim, which="SA")
        _X = np.real(EigVec)
        for i in i2n:
            emb_dic[i2n[i]] = torch.tensor(_X[i].tolist()).float().cpu()
        return emb_dic

