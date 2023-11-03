import os
import numpy as np
import torch
import pickle as pk

# this should be available
import utils.preprocess as prep

# SWaT dataset 
class SWaT_Dataset():
    def __init__(self, dataset_pth):
        self.dims = 51
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

        # make 5,10-th column as 0 ==> This is CRITICAL
        print('Make 5,10-th column as 0. This is critical.')
        dat['x_trn'][:, [5, 10]] = 0
        dat['x_tst'][:, [5, 10]] = 0

        return prep.window_stride(dat['x_trn'], dat['x_tst'], dat['lab_tst'], self.num_entity, dl, stride, tst_stride)

