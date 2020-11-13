#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Trainer script for src.pl_modules.supervised_learning. Example run command: bin/train.py save_to_folder configs/cnn.gin.
"""

import gin
from gin.config import _CONFIG
import torch
import logging
import os
import json
import collections

import torch
import pytorch_lightning as pl
import pandas as pd

logger = logging.getLogger(__name__)

from src.data import get_dataset
from src.utils import summary, acc, gin_wrap, parse_gin_config
from src.modules.supervised_training import SupervisedLearning
# Ensure gin seens all classes
from bin.train_supervised import *

import argh

def calculate_acc(weights, meta_data, config, train, valid, test):
    model = models.__dict__[config['train.model']]()
    model.load_state_dict(weights)
    pl_module = SupervisedLearning(model, meta_data=meta_data)

    trainer = pl.Trainer()
    results_test, = trainer.test(model=pl_module, test_dataloaders=test)
    results_valid, = trainer.test(model=pl_module, test_dataloaders=valid)
    results_train, = trainer.test(model=pl_module, test_dataloaders=train)
    monitor = 'valid_acc'
    return {'train': results_train[monitor], 'valid': results_valid[monitor], 'test': results_test[monitor]}

def combine_weights(weights_init, weights_final, config, step=0.5):
    train, valid, test, meta_data = get_dataset(batch_size=config['train.batch_size'], seed=config['train.seed'])

    results = {'freq': [], 'train':[], 'valid':[], 'test':[]}
    num_steps = int(1/step)
    print("step_size: {step}  | num_of_steps: {num_steps}")

    for i in range(num_steps+1):
        freq = i * step
        weights_temp = collections.OrderedDict()
        for k, v in weights_init.items():
            # it should add at the end, but i am not sure of that so i am calling it manually
            weights_temp[k] = (1-freq) * weights_init[k] + freq * weights_final[k]
            weights_temp.move_to_end(k)
        print("freq: {}". format(freq))
        results_step = calculate_acc(weights_temp, meta_data, config, train, valid, test)
        
        results['freq'].append(freq)
        for k,v in results_step.items():
            results[k].append(v)

    return results

def save_to_csv(dict, path, filename):
    df = pd.DataFrame(dict)
    df.to_csv(os.path.join(path, filename))


def evaluate(save_path, exp_name):
    print(save_path, exp_name)
    exp_path = os.path.join(save_path, exp_name)
    # Load config
    config = parse_gin_config(os.path.join(save_path, "config.gin"))
    gin.parse_config_files_and_bindings([os.path.join(os.path.join(save_path, "config.gin"))], bindings=[""])



    if not os.path.exists(exp_path):
        logger.info("Creating folder " + exp_path)
        os.system("mkdir -p " + exp_path)
    else:
        raise Error("There already exists a folder with this name")

    weights_init = torch.load(os.path.join(save_path, "initial0.pth"))
    weights_val = torch.load(os.path.join(save_path, "best_valid_acc49.pth"))
    weights_final = torch.load(os.path.join(save_path, "final61.pth"))
    

    results_val = combine_weights(weights_init, weights_val, config)
    save_to_csv(results_val, exp_path, "interpolation_val.csv")
    results_train = combine_weights(weights_init, weights_final, config)
    save_to_csv(results_train, exp_path, "interpolation_final.csv")

    # combine_weights(weights_init, weights_final, config)
    # with open(os.path.join(save_path, "eval_results_{}.json".format(checkpoint_name)), "w") as f:
    #     json.dump(results, f)


if __name__ == "__main__":
    argh.dispatch_command(evaluate)