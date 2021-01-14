import os

from PIL import Image
from easydict import EasyDict as edict
from torch.backends import cudnn
from config import config, default, dataset_config

from solvers import *
from data_loaders import *

default.network = 'MULTICYCLEGAN'
#default.network = 'STARGAN'
default.dataset_choice = ['MAKEUP']
#default.dataset_choice = ['CELEBA']
default.model_base = 'RES'
default.loss_chosen = 'normal'
default.gpu_ids = [0,1,2]

config_default = config


def train_net():
    # enable cudnn
    cudnn.benchmark = True

    # get the DataLoader
    data_loaders = eval("get_loader_" + config.network)(default.dataset_choice, dataset_config, config, mode="test")

    #get the solver
    solver = eval("Solver_" + config.network +"_VIS")(default.dataset_choice, data_loaders, config, dataset_config)
    solver.visualize()

if __name__ == '__main__':
    print("Call with args:")
    print(default)
    config = config_default[default.network]
    config.network = default.network
    config.model_base = default.model_base
    config.gpu_ids = default.gpu_ids

    # Create the directories if not exist
    if not os.path.exists(config.log_path):
        os.makedirs(config.log_path)
    if not os.path.exists(config.vis_path):
        os.makedirs(config.vis_path)
    if not os.path.exists(config.snapshot_path):
        os.makedirs(config.snapshot_path)
    if not os.path.exists(config.data_path):
        print("No datapath!!")

    train_net()
