import numpy as np
import pandas as pd
import datetime as dt
import pickle as pk

trn = pd.read_csv('SWaT_Dataset_Normal_v1.csv')
tst = pd.read_csv('SWaT_Dataset_Attack_v0.csv')

channels = trn.columns[1:-1]
x_trn = trn[trn.columns[1:-1]].to_numpy()
x_tst = tst[tst.columns[1:-1]].to_numpy()
lab_tst = tst['Normal/Attack'].to_numpy()
lab_tst[lab_tst == 'Normal'] = 0
lab_tst[lab_tst == 'Attack'] = 1
lab_tst[lab_tst == 'A ttack'] = 1
lab_tst = np.array(lab_tst, dtype = int)

with open('SWaT.pk', 'wb') as file:
    pk.dump({'channels': channels, 'x_trn': x_trn, 'x_tst': x_tst, 'lab_tst': lab_tst}, file)

print('done')
