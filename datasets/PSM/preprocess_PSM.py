import os
import numpy as np
import torch
import pickle as pk

# this should be available
import utils.preprocess as prep

# PSM dataset
class PSM_Dataset():
    def __init__(self, dataset_pth):
        self.dims = 25
        self.num_entity = 1
        with open(dataset_pth, 'rb') as file:
            self.dat = pk.load(file)

    def preprocess(self, params):
        # parameters
        dl = params.dl
        stride = params.stride
        tst_stride = dl if params.tst_stride == 'no_rep' else params.tst_stride

        # preprocess self.dat
        dat = prep.preprocess(self.dat, params, self.dims, self.num_entity, None)

        
        # output to csv file for usage
#         print('output to csv file for usage')
#         print(dat['x_trn'].shape, dat['x_tst'].shape, dat['lab_tst'].shape)
#         lab_trn = np.zeros(len(dat['x_trn']))
#         trn_cols = np.arange(dat['x_trn'].shape[-1] + 1)
#         trn_all = np.concatenate((dat['x_trn'], np.expand_dims(lab_trn, axis=-1)), axis=-1)
#         trn_all = np.concatenate((np.expand_dims(trn_cols, axis=0), trn_all), axis=0)
#         np.savetxt('PSM_down10_Train.csv', trn_all, delimiter=',')
# 
#         tst_cols = np.arange(dat['x_tst'].shape[-1] + 1)
#         tst_all = np.concatenate((dat['x_tst'], np.expand_dims(dat['lab_tst'], axis=-1)), axis=-1)
#         tst_all = np.concatenate((np.expand_dims(tst_cols, axis=0), tst_all), axis=0)
#         np.savetxt('PSM_down10_Test.csv', tst_all, delimiter=',')
#         exit()


        return prep.window_stride(dat['x_trn'], dat['x_tst'], dat['lab_tst'], self.num_entity, dl, stride, tst_stride)

