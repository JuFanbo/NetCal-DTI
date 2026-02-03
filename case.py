device = 'cuda'
import copy
from utils import *

class Case:
    def __init__(self, model, mode,obj=None):
        self.mode = mode
        if obj is None:
            obj = ['DB06334', 'P24941']
        resample()
        self.obj = obj
        self.dl = Loader('Data/case.csv')
        self.model = model
        self.case_data = None
        self.best_dl = None
        self.best_model = None

    def train(self):
        self.case_data_lst = [Case_data(i) for i in self.obj]
        batch_size = 256
        lr = 3e-4
        pt = 0
        best = 0
        criterion = CrossEntropyLoss()
        epochs = 500
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
            perf = self.performance(self.model, self.dl.vali, batch_size)
            print(f'Patience:{pt} Performance:{perf}')
            score = perf[-1]
            if score > best:
                self.best_model = copy.deepcopy(self.model)
                self.best_dl = copy.deepcopy(self.dl)
                best = score
                pt = 0
            else:
                pt += 1
                if pt > 7:
                    break
        torch.save(self.best_model.state_dict(), "state_dict.pth")
        for i in self.case_data_lst:
            i.update(model=self.best_model, dl=self.best_dl,mode=self.mode)
        print(f'Updated Case Data!')

    def performance(self, model, lst, batch_size):
        n_batch = (len(lst) + batch_size - 1) // batch_size
        Y = [i[2] for i in lst]
        model.eval()
        pred = []
        with torch.no_grad():
            for i in range(n_batch):
                start = i * batch_size
                end = min((i + 1) * batch_size, len(lst))
                emb1, emb2, seq, esm, smiles, mole, y = self.dl.prepare_data(lst[start:end])
                out = model(emb1, emb2, seq, esm, smiles, mole)
                out = torch.softmax(out.squeeze(), 1)[:, 1].squeeze()
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
        perf = [accuracy, recall, precision, F1, auc_score, aupr]
        perf = [round(i, 5) for i in perf]
        return perf
'''
for i in range(100):
    model = NetCalDTI().to('cuda')
    case=Case(model,'frequency',obj=['DB02546'])
    case.train()
'''