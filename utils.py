import random
from copy import copy

import numpy.random

from NetCalDTI import *
from dataloader import *
from torch.nn import CrossEntropyLoss
from sklearn.metrics import roc_auc_score,auc,precision_recall_curve,precision_score,recall_score,f1_score,accuracy_score
from tqdm import tqdm
def resample():
    lst = pd.read_csv('Data/drugbank.csv').values.tolist()
    lst = [i for i in lst if i[2] == 1]
    g = nx.Graph()
    for i in lst:
        g.add_edge(i[0],i[1])
    n = []
    ts = [i[0] for i in lst]
    ds = [i[1] for i in lst]
    for i in lst:
        tmp = []
        while len(tmp) == 0:
            d = random.choice(ds)
            if not d in list(g.neighbors(i[0])):
                tmp.append([i[0],d,0])
        n += tmp

        tmp = []
        while len(tmp) == 0:
            t = random.choice(ts)
            if not t in list(g.neighbors(i[1])):
                tmp.append([t, i[1], 0])
        n += tmp
    df = pd.DataFrame(lst + n)
    df.to_csv('Data/case.csv', index=False)

def id_mapping():
    mapping = {}
    for i in pd.read_csv('./Data/drug_info.csv').values.tolist():
        mapping[i[0]] = i[2]
    for j in pd.read_csv('./Data/target_mapping.csv').values.tolist():
        mapping[j[0]] = j[1]
    return mapping

def get_dict(x,model,dl:Loader):
    ds = pd.read_csv('./Data/drugbank.csv').values.tolist()
    targets = list(set([i[0] for i in ds]))
    drugs = list(set([i[1] for i in ds]))
    model.eval()
    batch_size = 256
    lst = [[x, i, 0] for i in drugs] if not 'DB' in x else [[i, x, 0] for i in targets]
    n_batch = (len(lst) + batch_size - 1) // batch_size
    dict = {}
    pred = []
    with torch.no_grad():
        for i in range(n_batch):
            start = i * batch_size
            end = min((i + 1) * batch_size, len(lst))
            emb1, emb2, seq, esm, smiles, mole, y = dl.prepare_data(lst[start:end])
            out = model(emb1, emb2, seq, esm, smiles, mole)
            out = torch.softmax(out.squeeze(), 1)[:, 1].squeeze()
            pred += out.detach().cpu().tolist()
    for i,j in zip(lst,pred):
        if 'DB' in x:
            dict[i[0]] = float(j)
        else:
            dict[i[1]] = float(j)
    return dict
class Training:
    def __init__(self, model,dti_path,k=128):
        self.dti_path = dti_path
        self.dl = Loader(dti_path,k=k)
        self.model = model

    def train(self,batch_size,lr,epochs=500,patience=10):

        pt = 0
        best = 0
        criterion = CrossEntropyLoss()
        epochs = epochs
        optimizer = torch.optim.Adam(params=self.model.parameters(), lr=lr)
        for _ in range(epochs):
            lst = self.dl.train
            random.shuffle(lst)
            n_batch = (len(lst) + batch_size - 1) // batch_size
            self.model.train()
            total_loss = 0
            batch_tqdm = tqdm(range(n_batch))
            for i in batch_tqdm:
                start = i * batch_size
                end = min((i + 1) * batch_size, len(lst))
                emb1, emb2, seq, esm, smiles, mole, y = self.dl.prepare_data(lst[start:end])
                out = self.model(emb1, emb2, seq, esm, smiles, mole).squeeze()
                loss = criterion(out, y)
                total_loss += loss.item()
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
            perf = self.performance(self.model, self.dl.vali,batch_size)
            print(f'Patience:{pt} Performance:{perf}')
            score = perf[-1]
            if score > best:
                best = score
                torch.save(self.model.state_dict(), 'state_dict.pth')
                with open('le.pickle', 'wb') as f:
                    pickle.dump(self.dl.network_embedding, f)
                pt = 0
            else:
                pt += 1
                if pt > patience:
                    break
        state_dict = torch.load('state_dict.pth',weights_only=True)
        self.model.load_state_dict(state_dict)
        with open('le.pickle', 'rb') as f:
            self.dl.network_embedding = pickle.load(f)
        perf = self.performance(self.model, self.dl.test,batch_size)
        print(f'***---{perf}---***')
        return perf

    def performance(self,model,lst,batch_size):
        n_batch = (len(lst) + batch_size - 1) // batch_size
        Y = [i[2] for i in lst]
        model.eval()
        pred = []
        with torch.no_grad():
            for i in range(n_batch):
                start = i * batch_size
                end = min((i + 1) * batch_size, len(lst))
                emb1,emb2,seq,esm,smiles,mole,y = self.dl.prepare_data(lst[start:end])
                out = model(emb1, emb2, seq, esm, smiles, mole)
                out = torch.softmax(out.squeeze(),1)[:,1].squeeze()
                out = out.detach().cpu().tolist()
                pred += out
        Y = np.array(Y)
        pred = np.array(pred)
        precision, recall, _ = precision_recall_curve(Y, pred, pos_label=1)
        aupr = float(auc(recall, precision))
        auc_score = roc_auc_score(Y, pred)
        pred = [int(round(i)) for i in pred]
        accuracy = accuracy_score(Y, pred)
        recall = recall_score(Y, pred)
        precision = precision_score(Y, pred)
        F1 = f1_score(Y, pred)
        perf = [accuracy,recall,precision,F1,auc_score,aupr]
        perf = [round(i,5) for i in perf]
        return perf

class Case_data:
    def __init__(self,x):
        self.mapping = id_mapping()
        self.count = 0
        self.x = x
        ds = pd.read_csv('./Data/case.csv').values.tolist()
        ds = [i for i in ds if i[2] == 1]
        self.neighbors = []
        if 'DB' in x:
            self.neighbors = [i[0] for i in ds if i[1] == x]
        else:
            self.neighbors = [i[1] for i in ds if i[0] == x]
        self.dict = {}
        if os.path.exists(f'{self.x}.csv'):
            self.read_from_csv()
        else:
            targets = list(set([i[0] for i in ds]))
            drugs = list(set([i[1] for i in ds]))
            if 'DB' in x:
                for i in targets:
                    self.dict[i] = 0
            else:
                for i in drugs:
                    self.dict[i] = 0

    def read_from_csv(self):
        dataframe = pd.read_csv(f'{self.x}.csv').values.tolist()
        self.count = int(dataframe[0][0])
        self.x = dataframe[0][1]
        for i in dataframe[1:]:
            self.dict[i[0]] = float(i[1])

    def update(self, model, dl: Loader, mode='direct'):
        '''
        Args:
        mode: 'direct' - direct addition mode
        'frequency' - frequency mode (add one to the top 10%)
        '''
        if self.count >= 100:
            print('Have Already Trained 100 Runs!!!')
            return
        else:
            self.count += 1
            new_dict = get_dict(self.x, model, dl)

            if mode == 'direct':
                for i in new_dict:
                    if i in self.dict:
                        self.dict[i] += float(new_dict[i])

            elif mode == 'frequency':
                all_scores = list(new_dict.values())
                all_scores.sort(reverse=True)
                threshold = all_scores[int(len(all_scores) * 0.1)]
                for i in self.dict:
                    if new_dict[i] >= threshold:
                        self.dict[i] += 1

            _lst = []
            tmp_lst = pd.read_csv(f'Data/drug_info.csv').values.tolist()
            dic = {i[0]: i[1] for i in tmp_lst}
            for i in self.dict:
                _lst.append([i, self.dict[i],
                             self.mapping[i] if i in self.mapping else '',
                             'yes' if i in self.neighbors else '',dic[i] if i in dic else ''])

            _lst.sort(key=lambda x: x[1], reverse=True)
            lst = [[self.count, self.x, None, None,None]] + _lst
            df = pd.DataFrame(lst)
            df.to_csv(f'{self.x}.csv', index=False)

def performance(model_name,_model,batch_size,lr,runs=5,patience=10):
    for dataset in ['drugbank','davis','bosnap','bindingdb']:
        result_lst = []
        for _ in range(runs):
            model = _model().to('cuda')
            t = Training(model=model, dti_path=f'Data/{dataset}.csv')
            result = t.train(batch_size=batch_size,lr=lr,patience=patience)
            result_lst.append(result)
            df = pd.DataFrame(result_lst)
            df.to_csv(f'{dataset}_{model_name}.csv', index=False)

