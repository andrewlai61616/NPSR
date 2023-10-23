import os
import datetime as dt
import numpy as np

import torch
from tqdm import tqdm

from utils.evaluation import *
from utils.utils import *

# when testing, try to find the maximum batch size to boost speed
def get_max_batch_size(trn_batch_size, device):
    mem_per_dat = torch.cuda.memory_reserved(device) / trn_batch_size
    total_mem = torch.cuda.get_device_properties(device).total_memory
    reserved_mem = torch.cuda.memory_reserved(device)
    _batch = int((total_mem - reserved_mem) / mem_per_dat)
    return min(1000, _batch)

# used when testing
def pass_test(_dset, _model, trn_batch_size, device):
    _batch = get_max_batch_size(trn_batch_size, device)
    with torch.no_grad():
        _out_all = []
        bn = np.arange(len(_dset))
        for i in range(int(np.ceil(len(_dset)/_batch))):
            _out_all.append(_model(_dset[bn[i*_batch:(i+1)*_batch]], sample=False).detach().cpu())
    return torch.cat(_out_all)


def train(data, models, optims, d_conf, m_conf, conf, eval_every = 1, seed=0):
    for model in models:
        model.notrain = False

    # extract training and testing inputs and outputs from [data]
    x_trn, y_trn, x_tst, y_tst, lab_tst = data['x_trn'], data['y_trn'], data['x_tst'], data['y_tst'], data['lab_tst']
    for mi in range(len(models)):
        x_trn[mi] = torch.from_numpy(x_trn[mi]).float().to(conf.device)
        y_trn[mi] = torch.from_numpy(y_trn[mi]).float().to(conf.device)
        x_tst[mi] = torch.from_numpy(x_tst[mi]).float().to(conf.device)
        
    # create path for saving model
    for i, model in enumerate(models):
        model.full_path = f'results/{d_conf.name}/{m_conf.name}/'
        model.full_fname = model.full_path + f'model{i}.pt'
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f'model {i}: {total_params} parameters')
    os.makedirs(model.full_path, exist_ok = True)

    # ensures the batches are sampled in the same sequence
    set_random_seed(seed)
    batch_size = m_conf.batch_size
    model.perfs = []
    best_perf = -1
    pbar = tqdm(range(m_conf.epochs), colour='blue', bar_format='{desc}|{bar}| {n_fmt}/{total_fmt} {elapsed}<{remaining}') # pbar.last_print_t - pbar.start_t

    # main training loop
    all_res = f'{d_conf}\n{m_conf}\n'
    for epoch in pbar:
        # train
        if epoch:
            trn_loss_mean = np.zeros(len(models))
            trn_loss_std = np.zeros(len(models))
            # loop over all models
            for mi in range(len(models)):
                trn_losses = []
                bn = np.arange(len(x_trn[mi]))
                np.random.shuffle(bn)

                for i in range(len(x_trn[mi]) // batch_size + (1 if len(x_trn[mi]) % batch_size else 0)):
                    X = x_trn[mi][bn[i*batch_size:(i+1)*batch_size]]
                    Y = y_trn[mi][bn[i*batch_size:(i+1)*batch_size]]
                    losses = ((models[mi](X) - Y) ** 2).mean(axis=-1)
                    loss = losses.mean()
                    optims[mi].zero_grad()
                    loss.backward()
                    optims[mi].step()
                    trn_losses.append(losses.detach().cpu().numpy())

                trn_losses = np.concatenate(trn_losses)
                trn_loss_mean[mi], trn_loss_std[mi] = trn_losses.mean(), trn_losses.std()
        else:
            # just evaluate in 0th epoch
            trn_loss_mean = np.ones(len(models)) * np.inf
            trn_loss_std = np.ones(len(models)) * np.inf

        # calculate reconstruction errors (tst_err)
        tst_loss_mean = np.zeros(len(models))
        tst_loss_std = np.zeros(len(models))
        tst_errs = []
        for mi in range(len(models)):
            y_tst_tilde = pass_test(
                _dset=x_tst[mi], _model=models[mi], trn_batch_size=batch_size, device=conf.device).numpy()
            tst_err = y_tst_tilde - y_tst[mi]
            tst_err = tst_err.reshape(-1, tst_err.shape[-1])
            if 'tst_use_pts' in data.keys():
                tst_err = tst_err[data['tst_use_pts'][mi]]
            tst_errs.append(tst_err)
            tst_loss_mean[mi] = (tst_err[lab_tst == 0] ** 2).mean(axis=-1).mean()
            tst_loss_std[mi] = (tst_err[lab_tst == 0] ** 2).mean(axis=-1).std()

# uncomment this section to look at M_pt, M_seq results
        # get M_pt, M_seq results
        eval_res = eval_epoch(lab_tst, tst_errs, conf)
        eval_stat = f'epoch={epoch}\n'
        for mi in range(len(models)):
            if hasattr(models[mi], 'model_name'):
                eval_stat += f'  {models[mi].model_name}: '
            else:
                eval_stat += f'  Model {mi}'

            eval_stat += f' trn loss = {trn_loss_mean[mi]:.5f}±{trn_loss_std[mi]:.5f}'
            eval_stat += f' tst loss = {tst_loss_mean[mi]:.5f}±{tst_loss_std[mi]:.5f}'
            eval_stat += f' F1:{eval_res[mi]["F1"]:.5f} AUC:{eval_res[mi]["AUC"]:.5f}'
            if mi < len(models)-1:
                eval_stat += '\n'

        # calculate the induced anomaly score
        if m_conf.name == 'NPSR':
            trn_errs = []
            for mi in range(len(models)):
                if mi == 0: # mi == 0 corresponds to M_pt
                    y_trn_tilde = pass_test(_dset=torch.from_numpy(data['x_trn_no_rep']).float().to(conf.device),
                        _model=models[mi], trn_batch_size=batch_size, device=conf.device).numpy()
                    trn_err = y_trn_tilde - data['x_trn_no_rep']
                else: # mi == 1 corresponds to M_seq
                    y_trn_tilde = pass_test(_dset=x_trn[mi], _model=models[mi], trn_batch_size=batch_size,
                                            device=conf.device).numpy()
                    trn_err = y_trn_tilde - y_trn[mi].cpu().numpy()
                trn_err = trn_err.reshape(-1, trn_err.shape[-1])

                # Discard the first and last [gamma] time points for M_pt (cf. supplementary B.2)
                if 'trn_use_pts' in data.keys():
                    trn_err = trn_err[data['trn_use_pts'][mi]]
                trn_errs.append(trn_err)

            # precalculate the train, test anomaly and nominality score, and theta_N
            trn_Nt = get_nominality_score(Delta_xp = trn_errs[0], Delta_x0 = trn_errs[1])
            tst_Nt = get_nominality_score(Delta_xp = tst_errs[0], Delta_x0 = tst_errs[1])
            trn_At = (trn_errs[0] ** 2).mean(axis=-1)
            tst_At = (tst_errs[0] ** 2).mean(axis=-1)
            theta_N = np.sort(trn_Nt)[int(len(trn_Nt) * m_conf.theta_N_ratio)]

            # similar to sec 3.4
            # calculate over a range of d = indc_len
            for indc_len in [1, 2, 4, 8, 16, 32, 64, 128, 256]:
                if data['num_entity'] == 1: # single entity
                    # calculates using soft, hard gate function, and given theta_N
                    indc_At_soft = get_induced_anomaly_score(tst_Nt, tst_At, theta_N, indc_len)
                    indc_At_hard = get_induced_anomaly_score(tst_Nt, tst_At, theta_N, indc_len, gate_func='hard')
                    indc_At_inf = get_induced_anomaly_score(tst_Nt, tst_At, np.inf, indc_len, gate_func='hard')
                    eval_stat += f'\n  '

                else:
                    indc_At_soft, indc_At_hard, indc_At_inf = [], [], []
                    for ei in range(data['num_entity']):
                        # partition trn_nominality_score, tst_nominality_score, anomaly_score
                        trn_Nt_ent = trn_Nt[data['trn_st'][ei]:data['trn_ed'][ei]] 
                        tst_Nt_ent = tst_Nt[data['tst_st'][ei]:data['tst_ed'][ei]]
                        tst_At_ent = tst_At[data['tst_st'][ei]:data['tst_ed'][ei]]
                        theta_N = np.sort(trn_Nt_ent)[int(len(trn_Nt_ent) * m_conf.theta_N_ratio)]
                        indc_At_soft.append(get_induced_anomaly_score(tst_Nt_ent, tst_At_ent, theta_N, indc_len))
                        indc_At_hard.append(get_induced_anomaly_score(tst_Nt_ent, tst_At_ent, theta_N, indc_len, 
                                                                      gate_func='hard'))
                        indc_At_inf.append(get_induced_anomaly_score(tst_Nt_ent, tst_At_ent, np.inf, indc_len,
                                                                     gate_func='hard'))
                    indc_At_soft = np.concatenate(indc_At_soft)
                    indc_At_hard = np.concatenate(indc_At_hard)
                    indc_At_inf = np.concatenate(indc_At_inf)
                    eval_stat += f'\n  '

                # calculate best F1 or F1pa score using the three induced anomaly scores
                if conf.eval_metric == 'bestF1':
                    NPSR_stat_soft = get_bestF1(lab_tst, indc_At_soft, PA=False)
                    NPSR_stat_hard = get_bestF1(lab_tst, indc_At_hard, PA=False)
                    NPSR_stat_inf = get_bestF1(lab_tst, indc_At_inf, PA=False)
                elif conf.eval_metric == 'bestF1pa':
                    NPSR_stat_soft = get_bestF1(lab_tst, indc_At_soft, PA=True)
                    NPSR_stat_hard = get_bestF1(lab_tst, indc_At_hard, PA=True)
                    NPSR_stat_inf = get_bestF1(lab_tst, indc_At_inf, PA=True)
                else:
                    assert False, f'eval_metric [{eval_metric}] undefined'

                eval_stat += f'd {indc_len:<5} Soft F1:{NPSR_stat_soft["F1"]:.5f}  AUC:{NPSR_stat_soft["AUC"]:.5f}  '
                eval_stat += f'Hard F1:{NPSR_stat_hard["F1"]:.5f}  AUC:{NPSR_stat_hard["AUC"]:.5f}  '
                eval_stat += f'Inf F1:{NPSR_stat_inf["F1"]:.5f}  AUC:{NPSR_stat_inf["AUC"]:.5f}'

        print(eval_stat)
        all_res += eval_stat + '\n'

#         print('threshold', theta_N)
#         np.save('results/L.npy', lab_tst)
#         np.save('results/N.npy', tst_Nt)
#         np.save('results/A.npy', tst_At)
#         exit()

    # save model
    # if train individual entities, don't save
    if 'entities' not in vars(d_conf) or d_conf.train_method != 'train_per_entity':
        print('Save trained model')
        for i, model in enumerate(models):
            torch.save(model.state_dict(), model.full_fname)
    # print output to text file with date
    today = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(f'{model.full_path}{today}_result.txt', 'a') as file:
        file.write(all_res + '\n')

