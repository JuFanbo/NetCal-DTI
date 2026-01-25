import pandas as pd
group=[]
value = []
lst = pd.read_csv('with_calibration/top100_novel.csv').values.tolist()
for i in lst:
    group.append("w/ calibration")
    value.append(i[5])
lst = pd.read_csv('without_calibration/top100_novel.csv').values.tolist()
for i in lst:
    group.append("w/o calibration")
    value.append(i[5])
df = pd.DataFrame({'group':group, 'value':value})
df.to_csv('comparison.csv', index=False)