folder = "with_calibration"
import pandas as pd
group = []
value = []
lst = pd.read_csv(f'{folder}/last100.csv').values.tolist()
for i in lst:
    group.append('last100')
    value.append(i[5])
lst = pd.read_csv(f'{folder}/top100.csv').values.tolist()
for i in lst:
    group.append('top100')
    value.append(i[5])
    group.append('known' if i[3] == 'yes' else 'novel')
    value.append(i[5])
df = pd.DataFrame({'group': group, 'value': value})
df.to_csv(f'{folder}/r_data.csv', index=False)