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

# def return_striding_content(arr, window, stride):
#     new_shape_row = (arr.shape[-1] - window)//stride + 1
#     new_shape_col = window
#     new_shape = (new_shape_row, new_shape_col)
#     n_bytes = arr.strides[-1]
#     stride_steps_row = n_bytes * stride
#     stride_step_col = n_bytes
#     stride_steps = (stride_steps_row, stride_step_col)
#     return np.lib.stride_tricks.as_strided(arr, new_shape, stride_steps)
# 
# class MGAB_Dataset():
#     def __init__(self, dataset_pth, relabel_anomaly_rad=None, only_rightside=False):
#         with open(dataset_pth, 'rb') as file:
#             self.dataset = pk.load(file)
#         self.num_entity = 1
# 
#         self.anomaly_rad = 200
#         self.anomaly_len = 401
#         lab_tst = self.dataset['lab_tst']
#         if relabel_anomaly_rad is not None:
#             # label [cut point - relabel_anomaly_len, cut point + relabel_anomaly_len] as anomaly
#             i = 0
#             while i < len(lab_tst):
#                 if lab_tst[i] == 1:
#                     lab_tst[i : i + self.anomaly_len] = 0
#                     if only_rightside:
#                         lab_tst[i + self.anomaly_rad : i + self.anomaly_rad + relabel_anomaly_rad + 1] = 1
#                     else:
#                         lab_tst[i + self.anomaly_rad - relabel_anomaly_rad : i + self.anomaly_rad + relabel_anomaly_rad + 1] = 1
#                     i += self.anomaly_rad + relabel_anomaly_rad * 2
#                 i += 1
#             self.anomaly_rad = relabel_anomaly_rad
#             self.anomaly_len = relabel_anomaly_rad * 2 + 1
#             if only_rightside:
#                 print('Only label at change point right side')
#             print(f'Redefined anomaly radius to {relabel_anomaly_rad}')
# 
#     def preprocess(self, params):
#         # parameters
#         dl = params.dl
#         stride = params.stride
#         tst_stride = dl // stride if params.tst_stride == 'no_rep' else params.tst_stride
# 
#         # Construct the sliding content
#         x_trn = return_striding_content(self.dataset['x_trn'], dl, stride)
#         x_tst = return_striding_content(self.dataset['x_tst'], dl, stride)
#         lab_tst = return_striding_content(self.dataset['lab_tst'], dl, stride)
# 
#         # apply stride/downsample to tst dataset
#         x_tst = x_tst[::tst_stride]
#         lab_tst = lab_tst[::tst_stride]
# 
#         # convert to torch Tensor
#         x_trn = torch.from_numpy(x_trn)
#         x_tst = torch.from_numpy(x_tst)
#         lab_tst = torch.from_numpy(lab_tst)
#         # make new dimension
#         x_trn = x_trn.unsqueeze(1)
#         x_tst = x_tst.unsqueeze(1)
#         return [x_trn, x_tst, lab_tst]

