import os
import numpy as np
import torch
import pickle as pk

# this should be available
import utils.preprocess as prep

# MSL dataset
class MSL_Dataset():
    def __init__(self, dataset_pth, entities):
        self.dims = 55
        with open(dataset_pth, 'rb') as file:
            self.dat = pk.load(file)
        if entities != 'all':
            print(entities)
            self.dat['x_trn'] = [self.dat['x_trn'][entity] for entity in entities]
            self.dat['x_tst'] = [self.dat['x_tst'][entity] for entity in entities]
            self.dat['lab_tst'] = [self.dat['lab_tst'][entity] for entity in entities]
            self.num_entity = len(entities)
        else:
            self.num_entity = 27

    def preprocess(self, params):
        # parameters
        dl = params.dl
        stride = params.stride
        tst_stride = dl if params.tst_stride == 'no_rep' else params.tst_stride

        if params.train_method == 'train_per_entity':
            if params.entity_id == self.num_entity:
                return None
            print(f'using entity {params.entity_id}/{self.num_entity-1}')
            dat = {}
            dat['x_trn'] = self.dat['x_trn'][params.entity_id]
            dat['x_tst'] = self.dat['x_tst'][params.entity_id]
            dat['lab_tst'] = self.dat['lab_tst'][params.entity_id]
            self.num_entity = 1
            dat = prep.preprocess(dat, params, self.dims, self.num_entity, None)
            return prep.window_stride(dat['x_trn'], dat['x_tst'], dat['lab_tst'], self.num_entity, dl, stride, tst_stride)
        else:
            # preprocess self.dat by each entity!
            x_trn_all, x_tst_all, lab_tst_all = [], [], []
            for entity_id in range(self.num_entity):
                dat_ent = {}
                for key in self.dat.keys():
                    dat_ent[key] = self.dat[key][entity_id]
                dat = prep.preprocess(dat_ent, params, self.dims, self.num_entity, entity_id, quiet=True)
                x_trn_all.append(dat['x_trn'])
                x_tst_all.append(dat['x_tst'])
                lab_tst_all.append(dat['lab_tst'])

            print('Add two additional channels for MSL (27 ent + 55 sens + 2 add = 84 = 12 heads * 7)')
            # Add two additional channels
            for ei in range(self.num_entity):
                x_trn_all[ei] = np.concatenate((np.zeros_like(x_trn_all[ei][:, :2]), x_trn_all[ei]), axis=-1)
                x_tst_all[ei] = np.concatenate((np.zeros_like(x_tst_all[ei][:, :2]), x_tst_all[ei]), axis=-1)

            return prep.window_stride(x_trn_all, x_tst_all, lab_tst_all, self.num_entity, dl, stride, tst_stride)
