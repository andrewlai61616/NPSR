import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

import torch
from sklearn.metrics import roc_auc_score
from scipy.stats import iqr

def get_bestF1(lab, scores, PA=False):
    scores = scores.numpy() if torch.is_tensor(scores) else scores
    lab = lab.numpy() if torch.is_tensor(lab) else lab
    ones = lab.sum()
    zeros = len(lab) - ones
    
    sortid = np.argsort(scores - lab * 1e-16)
    new_lab = lab[sortid]
    new_scores = scores[sortid]
    
    if PA:
        lab_diff = np.insert(lab, len(lab), 0) - np.insert(lab, 0, 0)
        a_st = np.arange(len(lab)+1)[lab_diff == 1]
        a_ed = np.arange(len(lab)+1)[lab_diff == -1]

        thres_a = np.array([np.max(scores[a_st[i]:a_ed[i]]) for i in range(len(a_st))])
        sort_a_id = np.flip(np.argsort(thres_a)) # big to small
        cum_a = np.cumsum(a_ed[sort_a_id] - a_st[sort_a_id])

        last_thres = np.inf
        TPs = np.zeros_like(new_lab)
        for i, a_id in enumerate(sort_a_id):
            TPs[(thres_a[a_id] <= new_scores) & (new_scores < last_thres)] = cum_a[i-1] if i > 0 else 0
            last_thres = thres_a[a_id]
        TPs[new_scores < last_thres] = cum_a[-1]
    else:
        TPs = np.cumsum(-new_lab) + ones
        
    FPs = np.cumsum(new_lab-1) + zeros
    FNs = ones - TPs
    TNs = zeros - FPs
    
    N = len(lab) - np.flip(TPs > 0).argmax()
    TPRs = TPs[:N] / ones
    PPVs = TPs[:N] / (TPs + FPs)[:N]
    FPRs = FPs[:N] / zeros
    F1s  = 2 * TPRs * PPVs / (TPRs + PPVs)
    maxid = np.argmax(F1s)
    
    FPRs = np.insert(FPRs, -1, 0)
    TPRs = np.insert(TPRs, -1, 0)
    if PA:
        AUC = ((TPRs[:-1] + TPRs[1:]) * (FPRs[:-1] - FPRs[1:])).sum() * 0.5
    else:
        AUC = roc_auc_score(lab, scores)
   
    anomaly_ratio = ones / len(lab) 
    FPR_bestF1_TPR1 = anomaly_ratio / (1-anomaly_ratio) * (2 / F1s[maxid] - 2)
    TPR_bestF1_FPR0 = F1s[maxid] / (2 - F1s[maxid])
    return {'AUC': AUC, 'F1': F1s[maxid], 'thres': new_scores[maxid], 'TPR': TPRs[maxid], 'PPV': PPVs[maxid], 
            'FPR': FPRs[maxid], 'maxid': maxid, 'FPRs': FPRs, 'TPRs': TPRs, 
            'FPR_bestF1_TPR1': FPR_bestF1_TPR1, 'TPR_bestF1_FPR0': TPR_bestF1_FPR0}

def eval_epoch(lab_tst, tst_errs, conf):
    if conf.score_function == 'Er':
        tst_Es = [(tst_err ** 2).mean(axis=-1) for tst_err in tst_errs]
    else:
        assert False, f'Score function must be Er; [{m_conf.score_function}] not implemented'
    
    if conf.eval_metric == 'bestF1':
        eval_res = [get_bestF1(lab_tst, tst_E, PA=False) for tst_E in tst_Es]
    elif conf.eval_metric == 'bestF1pa':
        eval_res = [get_bestF1(lab_tst, tst_E, PA=True) for tst_E in tst_Es]

    return eval_res


def get_nominality_score(Delta_xp, Delta_x0):
    assert len(Delta_xp.shape) == 2 and len(Delta_x0.shape) == 2
    assert Delta_xp.shape == Delta_x0.shape
    Delta_xd = Delta_x0 - Delta_xp
    nominality_score = (Delta_xd ** 2).mean(axis=-1) / (Delta_x0 ** 2).mean(axis=-1)
    return nominality_score


# note that for multi-entity datasets, only one entity should be input at a time
def get_induced_anomaly_score(nominality_score, anomaly_score, theta_N, d, gate_func = 'soft'):
    assert len(nominality_score.shape) == 1 and len(anomaly_score.shape) == 1
    assert nominality_score.shape == anomaly_score.shape

    if gate_func == 'soft':
        gN = 1 - nominality_score / theta_N
        gN[gN < 0] = 0
    elif gate_func == 'hard':
        gN = 1 - nominality_score / theta_N
        gN[gN < 0] = 0
        gN[gN > 0] = 1
    else:
        assert False, f'gate function [{gate_func}] not defined'
    induced_anomaly_score = np.copy(anomaly_score)

    # calculate e_t - s_t + 1
    denom = np.ones(len(gN)) * np.min((len(gN), 2*d+1)) # max total induction length
    if d < len(gN) - 1:
        denom[:d] = np.min((denom[:d], np.arange(d+1, 2*d+1)), axis=0)
        denom[-1:-d-1:-1] = np.min((denom[-1:-d-1:-1], np.arange(d+1, 2*d+1)), axis=0)

    # calculate induced anomaly score
    gN_forw = np.copy(sliding_window_view(np.concatenate((gN, np.zeros(d-1))), len(gN)))
    gN_back = np.flip(np.copy(sliding_window_view(np.concatenate((np.zeros(d-1), gN)), len(gN))), axis=0)
    gN_forw = np.cumprod(gN_forw[:, 1:], axis=0)
    gN_back = np.cumprod(gN_back[:, :-1], axis=0)
    A_gN_forw_flip = np.flip(np.expand_dims(anomaly_score[:-1], axis=0) * gN_forw, axis=-1)
    A_gN_back = np.expand_dims(anomaly_score[1:], axis=0) * gN_back
    
    numer = np.insert(np.flip([np.diagonal(A_gN_forw_flip, i).sum() for i in range(len(gN)-1)]), 0, 0)
    numer[:-1] += np.array([np.diagonal(A_gN_back, i).sum() for i in range(len(gN)-1)])
    induced_anomaly_score += numer * 2 * d / denom
    return induced_anomaly_score

