import sys
import numpy as np
from MGAB import mgab

args = {
    'verbosity': 1,
    'output_dir': None,
    'output_force_override': False,
    'output_format': 'csv',
    'num_series': 1,
    'series_length': 1000000,
    'num_anomalies': 0,
    'noise': 'rnd_uniform',
    'noise_param': (-0.01, 0.01),
    'min_anomaly_distance': 2000,
    'mg_ts_path_load': None,
    'mg_tau': 18.0,
    'mg_n': 10.0,
    'mg_beta': 0.25,
    'mg_gamma': 0.1,
    'mg_history': 0.9,
    'mg_T': 1.0,
    'mg_ts_dir_save': None,
    'seed': None,
    'min_length_cut': 100,
    'max_sgmt': 100,
    'anomaly_window': 50,
    'order_derivative': 3,
    'reproduce_original_mgab': None
}
x_trn = mgab.generate_benchmark(args)[0]
assert x_trn['is_anomaly'].sum() == 0
x_trn = np.expand_dims(x_trn['value'].to_numpy(), axis=-1)
print('x_trn', x_trn.shape)

anomaly_ratio = 0.01
args['series_length'] = 200000
args['seed'] = 1
args['num_anomalies'] = int(args['series_length'] * anomaly_ratio // args['anomaly_window'])
print('num anomalies =', args['num_anomalies'])
x_tst = mgab.generate_benchmark(args)[0]

lab_tst = x_tst['is_anomaly'].to_numpy()
x_tst = np.expand_dims(x_tst['value'].to_numpy(), axis=-1)

print('x_tst', x_tst.shape)
print('lab_tst', lab_tst.shape)
print(f'There are {lab_tst.sum()} anomaly points')

import pickle as pk
with open('MGAB.pk', 'wb') as file:
    pk.dump({'x_trn': x_trn, 'x_tst': x_tst, 'lab_tst': lab_tst}, file)

print('Construction done')
