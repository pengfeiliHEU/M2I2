import os
import torch
import numpy as np
import random
import torch.backends.cudnn as cudnn


# 设置随机种子
def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True
    print("set random seed: ", seed)


def make_dirs(path: str):
    if not os.path.exists(path):
        try:
            os.makedirs(path)
        except OSError as exc:
            if exc.errno != errno.EEXIST:
                raise


def replace_dirname(path: str):
    old_name = path
    new_name = path.replace('(running)', '')
    os.rename(old_name, new_name)

