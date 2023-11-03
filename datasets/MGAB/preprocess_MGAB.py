import numpy as np
import torch
import pickle as pk

# this should be available
import utils.preprocess as prep

# https://github.com/MarkusThill/MGAB
# The Mackey-Glass Anomaly Benchmark
class MGAB_Dataset():
    def __init__(self, dataset_pth):
        self.dims = 1
        self.num_entity = 1
        with open(dataset_pth, 'rb') as file:
            self.dat = pk.load(file)

        # Performer won't work if dims = 1
        print('Expanding data to 2 dims so that Performer can work')
        self.dat['x_trn'] = np.repeat(self.dat['x_trn'], 4, axis=-1)
        self.dat['x_tst'] = np.repeat(self.dat['x_tst'], 4, axis=-1)
        self.dims = 4

    def preprocess(self, params):
        # parameters
        dl = params.dl
        stride = params.stride
        tst_stride = dl if params.tst_stride == 'no_rep' else params.tst_stride

        # preprocess self.dat
        dat = prep.preprocess(self.dat, params, self.dims, self.num_entity, None)

        return prep.window_stride(dat['x_trn'], dat['x_tst'], dat['lab_tst'], self.num_entity, dl, stride, tst_stride)
