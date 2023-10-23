import os
import numpy as np
import pandas as pd
import pickle as pk
import matplotlib.pyplot as plt

datapath = 'TSADIS/MSCRED.csv'

dat = pd.read_csv(datapath, index_col=0)
dat_gt = pd.read_csv('TSADIS/MSCRED_GT.csv', index_col=0)

# dataset split according to https://arxiv.org/abs/1811.08055
trn_tst_split_point = 10000

x_trn = dat[:trn_tst_split_point].values
x_tst_cont = dat[trn_tst_split_point:].values
lab_tst_cont = (dat_gt[trn_tst_split_point:].values.sum(axis=1) > 0) * 1
x_tst = []
lab_tst = []

# # extract test sections
tst_sections = [[0, 1720], [1900, 2850], [4630, 7700], [7880, 8530], [8710, 10000]]
errors = 0
num_points = 0
for sec in tst_sections:
    errors += lab_tst_cont[sec[0]:sec[1]].sum()
    num_points += sec[1] - sec[0]
    x_tst.append(x_tst_cont[sec[0]:sec[1]])
    lab_tst.append(lab_tst_cont[sec[0]:sec[1]])
anomaly_rate = errors / num_points
print('anomaly rate =', anomaly_rate)
x_trn = [x_trn] * len(tst_sections)

# save dataset to pickle file
MSCRED_path = 'MSCRED.pk'
with open(MSCRED_path, 'wb') as pkf:
    pk.dump({'x_trn': x_trn, 'x_tst': x_tst, 'lab_tst': lab_tst}, pkf)