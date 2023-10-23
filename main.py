# Script for running NPSR on multiple datasets
import sys
import black
from argparse import Namespace

import torch

from utils.utils import *
from utils import datasets
from train import train

def main():
    # loop through datasets
    for d_conf in config.dsets:
        m_confs = d_conf.models
        del d_conf.models
        print('[DATASET]', d_conf)

        # loop through algorithms (currently only implemented NPSR)
        for m_conf in m_confs:
            if 'entities' in vars(d_conf) and d_conf.train_method == 'train_per_entity':
                d_conf.entity_id = 0
            while True:
                # retrieve dataset
                data = datasets.get_dataset_processed(d_conf)
                
                if data is None: # all entities have been trained
                    break
                print('[MODEL]', m_conf)
                # construct model
                models = get_model(m_conf, d_conf, data, seed=0)
                for model in models['models']:
                    model = model.to(device)
                    model.m_conf, model.d_conf = m_conf, d_conf
                # train the model
                train(data, models['models'], models['opts'], d_conf, m_conf, config)
                if 'entities' in vars(d_conf) and d_conf.train_method == 'train_per_entity':
                    d_conf.entity_id += 1
                else:
                    break

if __name__ == '__main__':
    # parse arguments
    assert len(sys.argv) == 2, 'Usage: python main.py config.txt'
    config = parse_config(sys.argv[1])
    config_str = black.format_str(f'{config}', mode=black.FileMode())

    # initialization
    device = torch.device(config.device)
    print('Using device:', device, file=sys.stderr)
    print('Total memory:', torch.cuda.get_device_properties(device).total_memory)

    # main procedure
    main()
