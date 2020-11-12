#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
See main.py -h for help.
"""
import numpy as np
import argh
import glob
import os
import matplotlib.pylab as plt
from os.path import dirname, basename, join

import pandas as pd

from gin.config import _CONFIG
from src import RESULTS_DIR
from src.utils import load_C, load_H, load_HC
# This will be important for passing gin configs
from bin.train_supervised import *
from src import *

EXPERIMENT_DIR = dirname(__file__)
logger = logging.getLogger(__name__)

def prepare(experiment="large"):
    os.system("mkdir -p {}/configs".format(experiment))
    if experiment == "large":
        lrs = [0.001, 0.01, 0.1]
    elif experiment == "small":
        lrs = [0.001, 0.1]
    else:
        raise NotImplementedError()

    # Write configs by modifying template
    with open(join(EXPERIMENT_DIR, "template_config.gin"), "r") as f:
        TEMPLATE = f.read()
    for id, bs in enumerate(lrs):
        exp_config_path = join(EXPERIMENT_DIR, experiment, "configs", "{}.gin".format(id))
        os.system("mkdir -p " + dirname(exp_config_path))
        with open(exp_config_path, "w") as c:
            c.write(TEMPLATE.replace("$learning_rate$", str(bs)))

    # Prepare batch of runs
    with open(join(EXPERIMENT_DIR, experiment, "batch.sh"), "w") as f:
        for id in range(len(lrs)):
            exp_save_path = join(RESULTS_DIR, "tune_lr", experiment, str(id))
            exp_config_path = join(EXPERIMENT_DIR, experiment, "configs", "{}.gin".format(id))
            if not os.path.exists(join(exp_save_path, "FINISHED")):
                os.system("mkdir -p " + exp_save_path)
                f.write("python3 bin/train_supervised.py {save_path} {config_path} \n".format(
                    save_path=exp_save_path,
                    config_path=exp_config_path
                ))
            else:
                logger.info("Finished experiment #{}, checking if configs match.".format(id))
                # Ensures that run experiment matches expectations
                with open(exp_config_path, "r") as f_c:
                    with open(join(exp_save_path, "{}.gin".format(id))) as f_e:
                        c = f_c.read()
                        c_run = f_e.read()
                        assert c == c_run, "Finished experiment with a mismatching config. Aborting."


def report(experiment="large"):
    SAVE_TO = join(EXPERIMENT_DIR, experiment)

    Es = glob.glob(join(RESULTS_DIR, "tune_lr", experiment, "*"))
    Es = sorted(Es, key=lambda e: int(basename(e)))
    Es = [E for E in Es if os.path.exists(join(E, 'FINISHED'))] # Use only finished

    x, y = [], []
    for E in Es:
        H, C = load_HC(E)
        lr = C["LRSchedule.base_lr"]
        x.append(lr)
        y.append(max(H['valid_acc']))

    logger.info("Maximum accuracy reached for learning_rate={}.".format(x[np.argmax(y)]))

    plt.plot(x, y)
    plt.xlabel("Learning rate")
    plt.ylabel("Maximum accuracy")
    plt.savefig(join(SAVE_TO, 'plot.pdf'))
    plt.show()

if __name__ == "__main__":
    argh.dispatch_commands([prepare, report])
