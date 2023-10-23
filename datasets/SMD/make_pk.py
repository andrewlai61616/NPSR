import os
import pickle as pk

import numpy as np
import pandas as pd

pth = './'

x_trn, x_tst, lab_tst = [], [], []
for ent_name in os.listdir(pth + 'train'):
    x_trn.append(pd.read_csv(pth + 'train/' + ent_name, header=None).to_numpy())
    x_tst.append(pd.read_csv(pth + 'test/' + ent_name, header=None).to_numpy())
    lab_tst.append(np.squeeze(pd.read_csv(pth + 'test_label/' + ent_name, header=None).to_numpy()))
    
print('Dumping pickle files...')
with open(pth + 'SMD.pk', 'wb') as file:
    pk.dump({'x_trn': x_trn, 'x_tst': x_tst, 'lab_tst': lab_tst}, file)

