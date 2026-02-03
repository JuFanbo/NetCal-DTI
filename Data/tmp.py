import pandas as pd
lst = pd.read_csv('bindingdb.csv').values.tolist()
targets = [i[0] for i in lst]
drugs = [i[1] for i in lst]
print(len([i for i in lst if i[2]== 1]))
print(len([i for i in lst if i[2] == 0]))