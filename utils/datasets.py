"""
Dataset handler:

MGAB
SWaT
WADI
SMAP
MSL
SMD
PSM

"""

import sys

def get_dataset_processed(params):
    if params.name == 'MGAB':
        data_path = 'datasets/MGAB/'
        if data_path not in sys.path:
            sys.path.append(data_path)
        from preprocess_MGAB import MGAB_Dataset
        dataset = MGAB_Dataset(dataset_pth = data_path + params.name + '.pk')

    elif params.name == 'SWaT':
        data_path = 'datasets/SWaT/'
        if data_path not in sys.path:
            sys.path.append(data_path)
        from preprocess_SWaT import SWaT_Dataset
        dataset = SWaT_Dataset(dataset_pth = data_path + params.name + '.pk')

    elif params.name == 'WADI':
        data_path = 'datasets/WADI/'
        if data_path not in sys.path:
            sys.path.append(data_path)
        from preprocess_WADI import WADI_Dataset
        dataset = WADI_Dataset(dataset_pth = data_path + params.name + '.pk')

    elif params.name == 'SMAP':
        data_path = 'datasets/SMAP/'
        if data_path not in sys.path:
            sys.path.append(data_path)
        from preprocess_SMAP import SMAP_Dataset
        dataset = SMAP_Dataset(dataset_pth = data_path + params.name + '.pk', entities = params.entities)

    elif params.name == 'MSL':
        data_path = 'datasets/MSL/'
        if data_path not in sys.path:
            sys.path.append(data_path)
        from preprocess_MSL import MSL_Dataset
        dataset = MSL_Dataset(dataset_pth = data_path + params.name + '.pk', entities = params.entities)

    elif params.name == 'SMD':
        data_path = 'datasets/SMD/'
        if data_path not in sys.path:
            sys.path.append(data_path)
        from preprocess_SMD import SMD_Dataset
        dataset = SMD_Dataset(dataset_pth = data_path + params.name + '.pk', entities = params.entities)

    elif params.name == 'PSM':
        data_path = 'datasets/PSM/'
        if data_path not in sys.path:
            sys.path.append(data_path)
        from preprocess_PSM import PSM_Dataset
        dataset = PSM_Dataset(dataset_pth = data_path + params.name + '.pk')

    elif params.name == 'trimSyn':
        data_path = 'datasets/trimSyn/'
        if data_path not in sys.path:
            sys.path.append(data_path)
        from preprocess_trimSyn import trimSyn_Dataset
        dataset = trimSyn_Dataset(dataset_pth = data_path + params.name + '.pk', entities = params.entities)
        
    else:
        print('Cannot find dataset name!')
        raise

    return dataset.preprocess(params)
