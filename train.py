import os

import argparse
import random

import numpy as np
import torch
from torch.backends import cudnn

from dataloder import get_loader
from psgan.solver import Solver
from setup import setup_config, setup_argparser


def train_net(config):
    # enable cudnn  https://zhuanlan.zhihu.com/p/73711222
    cudnn.benchmark = True

    data_loader = get_loader(config)
    # solver = Solver(config, data_loader=data_loader, device="cuda")
    solver = Solver(config, data_loader=data_loader, device=config.device)
    solver.train()


if __name__ == '__main__':
    args = setup_argparser().parse_args()
    config = setup_config(args)
    print("Call with args:")
    print(config)

    seed = 0
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    train_net(config)

# lib pip install
# 1.requests
# 2.matplotlib
# 3.fvcore
# 4.dlib==19.6.1
