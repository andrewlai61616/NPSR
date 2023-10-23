import random
import numpy as np
from argparse import Namespace

import torch
from torch.nn import functional as F

from models import NPSR

def parse_value(value: str):
    # may be int, float, str, bool
    try:
        value = float(value)
        return int(value) if np.round(value) == value else value
    except ValueError:
        pass
    return True if value == 'True' else (False if value == 'False' else (None if value == 'None' else value))

def parse_bool(value: str):
    if value == 'True':
        return True
    elif value == 'False':
        return False
    print(f'Parse value not True/False! value={value}')
    raise

def parse_int(value: str):
    if value == 'None':
        return None
    return int(value)

def set_random_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)

def return_striding_content(arr, window, stride):
    new_shape_row = (arr.shape[-1] - window)//stride + 1
    new_shape_col = window
    new_shape = (new_shape_row, new_shape_col)
    n_bytes = arr.strides[-1]
    stride_steps_row = n_bytes * stride
    stride_step_col = n_bytes
    stride_steps = (stride_steps_row, stride_step_col)
    return np.lib.stride_tricks.as_strided(arr, new_shape, stride_steps)

def parse_config(filename):
    config = Namespace()
    dsets = []
    default_dataset_config = Namespace()
    default_model_trn_config = {}

    section = 'general'
    fp = open(filename)
    lines = fp.readlines()
    for line in lines:
        # remove comments
        line = line.replace('\n', '').split('#')[0].strip()
        if len(line) == 0 or line[0] == '':
            continue

        # section/dataset/model definition strings
        if line.split(' ')[0] =='default_dataset_config':
            section = 'default_dataset_config'
            continue
        elif line.split(' ')[0] == 'default_model_trn_config':
            section = 'default_model_trn_config'
            continue
        elif line.split(' ')[0] == 'dset_model_trn_config':
            section = 'dset_model_trn_config'
            continue
        elif line.split(' ')[0] == 'dataset':
            dataset = line.split(' ')[-1]
            # apply default dataset config
            dsets.append(Namespace())
            dsets[-1].name = dataset
            dsets[-1] = Namespace(**vars(dsets[-1]), **vars(default_dataset_config))
            dsets[-1].models = []
            model = None
            continue
        elif line.split(' ')[0] =='model':
            model = line.split(' ')[-1]
            if model not in default_model_trn_config.keys():
                default_model_trn_config[model] = Namespace()
            if section == 'default_model_trn_config':
                pass
            elif section == 'dset_model_trn_config':
                dsets[-1].models.append(Namespace())
                dsets[-1].models[-1].name = model
                dsets[-1].models[-1] = Namespace(**vars(dsets[-1].models[-1]),
                                                 **vars(default_model_trn_config[model]))
            continue

        # get key and value
        key = line.split(' ')[0]
        value = line.split(' ')[-1]

        # insert into configs
        if section == 'general':
            setattr(config, key, parse_value(value))

        elif section == 'default_dataset_config' or (section == 'dset_model_trn_config' and model == None):
            tmpns = Namespace()
            if key == 'entities':
                if value == 'all':
                    tmpns.entities = value
                else:
                    values = line[len(key)+1:].replace(' ', '')[1:-1].split(',')
                    tmpns.entities = [int(v) for v in values]
            else:
                setattr(tmpns, key, parse_value(value))

            if section == 'default_dataset_config':
                default_dataset_config = Namespace(**vars(default_dataset_config), **vars(tmpns))
            else:
                new_ns: dict = vars(dsets[-1])
                new_ns.update(vars(tmpns))
                dsets[-1] = Namespace(**new_ns)

        elif section == 'dset_model_trn_config' or section == 'default_model_trn_config':
            tmpns = Namespace()
            setattr(tmpns, key, parse_value(value))

            if section == 'default_model_trn_config':
                default_model_trn_config[model] = Namespace(**vars(default_model_trn_config[model]),
                                                            **vars(tmpns))
            else:
                new_ns: dict = vars(dsets[-1].models[-1])
                new_ns.update(vars(tmpns))
                dsets[-1].models[-1] = Namespace(**new_ns)
        else:
            print(f'Invalid section {section}')

    fp.close()
    config.dsets = dsets

    return config

def get_model(m_conf, d_conf, data, seed=0):
    x_trn, x_tst, lab_tst = data['x_trn'], data['x_tst'], data['lab_tst']
    length, channel = x_trn.shape[1], x_trn.shape[-1]

    # ensures the model is initialized at the same state
    set_random_seed(seed)

    if m_conf.name == 'NPSR':
        assert d_conf.tst_stride == 'no_rep', 'tst_stride must be \"no_rep\" in order to use NPSR model'
        delta, pred_dl = m_conf.delta, m_conf.pred_dl
        delta_dl = pred_dl + delta
        trn, tst, lab = data['trn'], data['tst'], data['lab']
        x_trn_cut, y_trn_cut, x_tst_cut, y_tst_cut, lab_c = [], [], [], [], []

        # construct M_pt and its optimizer
        model_rec = NPSR.PerformerAEPositionalEncoding(W=d_conf.dl, D=channel,
                        heads=m_conf.heads, dep=m_conf.enc_depth, lat=m_conf.z_dim,
                        ff_mult=m_conf.ff_mult)
        rec_opt = torch.optim.Adam(model_rec.parameters(), lr=m_conf.learn_rate)

        # construct M_seq and its optimizer
        model_pred = NPSR.PerfPredSqz(Win=pred_dl, Wout=m_conf.delta, D=channel,
                        heads=m_conf.heads, dep=m_conf.pred_depth, ff_mult=m_conf.ff_mult)
        pred_opt = torch.optim.Adam(model_pred.parameters(), lr=m_conf.learn_rate)

        # Construct x_trn, y_trn, x_tst, y_tst for prediction model 
        if data['num_entity'] == 1:
            # Cut data to correct length
            # construct trn data
            for si in range(0, len(trn)-delta_dl+1, delta):
                x_trn_cut.append(trn[si : si+delta_dl])
                y_trn_cut.append(trn[si+pred_dl//2 : si+pred_dl//2+delta])
            x_trn_cut, y_trn_cut = np.stack(x_trn_cut), np.stack(y_trn_cut)
            x_trn_cut = np.concatenate((x_trn_cut[:, :pred_dl//2], x_trn_cut[:, -pred_dl//2:]), axis=1)
            # construct tst data
            for si in range(0, len(tst)-delta_dl+1, delta):
                x_tst_cut.append(tst[si : si+delta_dl])
                y_tst_cut.append(tst[si+pred_dl//2 : si+pred_dl//2+delta])
            x_tst_cut, y_tst_cut = np.stack(x_tst_cut), np.stack(y_tst_cut)
            x_tst_cut = np.concatenate((x_tst_cut[:, :pred_dl//2], x_tst_cut[:, -pred_dl//2:]), axis=1)
            lab_c = lab[pred_dl//2 : len(lab) - (len(lab)-pred_dl)%delta - pred_dl//2]

            trn_use_pts = [np.arange(pred_dl//2, len(trn) - (len(trn)-pred_dl)%delta - pred_dl//2),
                           np.arange(len(trn) - (len(trn)-pred_dl)%delta - pred_dl)]
            tst_use_pts = [np.arange(pred_dl//2, len(lab) - (len(lab)-pred_dl)%delta - pred_dl//2),
                           np.arange(len(lab_c))]

            # construct no rep x_trn
            x_trn_no_rep = np.concatenate((trn, trn[:d_conf.dl - len(trn)%d_conf.dl]), axis=0).reshape(-1, d_conf.dl, channel)
        else:
            # construct trn data
            for ei in range(data['num_entity']):
                st = 0
                while st + delta_dl <= len(trn[ei]):
                    x_trn_cut.append(trn[ei][st:st+delta_dl])
                    y_trn_cut.append(trn[ei][st+pred_dl//2:st+pred_dl//2+delta])
                    st += delta
            x_trn_cut, y_trn_cut = np.stack(x_trn_cut), np.stack(y_trn_cut)
            x_trn_cut = np.concatenate((x_trn_cut[:, :pred_dl//2], x_trn_cut[:, -pred_dl//2:]), axis=1)

            # construct tst data
            for ei in range(data['num_entity']):
                st = 0
                while st + delta_dl <= len(tst[ei]):
                    x_tst_cut.append(tst[ei][st:st+delta_dl])
                    y_tst_cut.append(tst[ei][st+pred_dl//2:st+pred_dl//2+delta])
                    lab_c.append(lab[ei][st+pred_dl//2:st+pred_dl//2+delta])
                    st += delta
            x_tst_cut, y_tst_cut = np.stack(x_tst_cut), np.stack(y_tst_cut)
            x_tst_cut = np.concatenate((x_tst_cut[:, :pred_dl//2], x_tst_cut[:, -pred_dl//2:]), axis=1)
            lab_c = np.concatenate(lab_c)

            # calculate start and end points for each entity
            trn_ed = np.cumsum(np.array([len(ent) for ent in trn]))
            trn_st = np.insert(trn_ed[:-1], 0, 0)
            tst_ed = np.cumsum(np.array([len(ent) for ent in tst]))
            tst_st = np.insert(tst_ed[:-1], 0, 0)
            trn_st_pad = trn_st + pred_dl//2
            trn_ed_pad = trn_ed - pred_dl//2 - (trn_ed - pred_dl//2 - trn_st_pad) % delta
            tst_st_pad = tst_st + pred_dl//2
            tst_ed_pad = tst_ed - pred_dl//2 - (tst_ed - pred_dl//2 - tst_st_pad) % delta

            # this is for cutting the start and end [gamma] points (cf. supplementary B.2)
            # calculate the 'indices' that needs to be retained
            trn_use_pts = [ np.concatenate( [ np.arange(trn_st_pad[ei], trn_ed_pad[ei])
                for ei in range(data['num_entity'])]), np.arange(y_trn_cut.shape[0] * delta) ]
            tst_use_pts = [ np.concatenate( [ np.arange(tst_st_pad[ei], tst_ed_pad[ei])
                for ei in range(data['num_entity'])]), np.arange(len(lab_c)) ]

            # construct no rep x_trn
            trn_cat = np.concatenate(trn)
            trn_cat = np.concatenate((trn_cat, trn_cat[:d_conf.dl - len(trn_cat)%d_conf.dl]), axis=0)
            x_trn_no_rep = trn_cat.reshape(-1, d_conf.dl, channel)

            # because all entities are aggregated together in the combined method for multi-entity datasets.
            # we need to specify the start and end time points for each entity in the concatenated data
            data['trn_st'], data['trn_ed'], data['tst_st'], data['tst_ed'] = [], [], [], []
            for ei in range(data['num_entity']):
                trn_ent_len = len(trn[ei]) - (len(trn[ei])-pred_dl)%delta - pred_dl
                data['trn_st'].append(0 if len(data['trn_st']) == 0 else data['trn_ed'][-1])
                data['trn_ed'].append(data['trn_st'][-1] + trn_ent_len)
                tst_ent_len = len(tst[ei]) - (len(tst[ei])-pred_dl)%delta - pred_dl
                data['tst_st'].append(0 if len(data['tst_st']) == 0 else data['tst_ed'][-1])
                data['tst_ed'].append(data['tst_st'][-1] + tst_ent_len)

        data['x_trn'], data['y_trn'] = [x_trn, x_trn_cut], [x_trn, y_trn_cut]
        data['x_tst'], data['y_tst'] = [x_tst, x_tst_cut], [x_tst, y_tst_cut]
        data['lab_tst'] = lab_c
        data['trn_use_pts'] = trn_use_pts
        data['tst_use_pts'] = tst_use_pts
        data['x_trn_no_rep'] = x_trn_no_rep
        return {'models': [model_rec, model_pred], 'opts': [rec_opt, pred_opt]}
        
    assert False, f'Model [{m_conf.name}] not defined'



