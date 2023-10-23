import os
import pickle as pk

import numpy as np
import pandas as pd

trn = pd.read_csv(f'train.csv')
tst = pd.read_csv(f'test.csv')
lab = pd.read_csv(f'test_label.csv')

channels = trn.columns[1:]
x_trn = trn.fillna(method='ffill').to_numpy()[:, 1:]
x_tst = tst.fillna(method='ffill').to_numpy()[:, 1:]
lab_tst = lab['label'].to_numpy()

with open('PSM.pk', 'wb') as file:
    pk.dump({'channels': channels, 'x_trn': x_trn, 'x_tst': x_tst, 'lab_tst': lab_tst}, file)

print('done')
