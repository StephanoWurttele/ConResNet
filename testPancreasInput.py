import argparse
import sys
sys.path.append("..")

import torch
import numpy as np
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.functional as F

import os
import os.path as osp
from models.ConResNet import ConResNet
from dataset.PancreasDataSet import PancreasDataSet, PancreasValDataSet
import timeit
# from tensorboardX import SummaryWriter
from utils import loss
# from utils.engine import Engine
from math import ceil

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
        
def get_arguments():
    """
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="ConResNet for 3D Medical Image Segmentation.")

    parser.add_argument("--data_dir", type=str, default='/media/userdisk0/myproject-Seg/BraTS-pro/dataset/')
    parser.add_argument("--train_list", type=str, default='list/BraTS2018_old/train.txt')
    parser.add_argument("--val_list", type=str, default='list/BraTS2018_old/val.txt')
    parser.add_argument("--snapshot_dir", type=str, default='snapshots/conresnet/')
    parser.add_argument("--input_size", type=str, default='80,160,160')
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_gpus", type=int, default=1)
    parser.add_argument("--num_steps", type=int, default=40000)
    parser.add_argument("--val_pred_every", type=int, default=100)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--num_classes", type=int, default=3)
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--random_mirror", type=str2bool, default=False)
    parser.add_argument("--random_scale", type=str2bool, default=False)

    return parser

parser = get_arguments()
print(parser)
args = parser.parse_args()
d, h, w = map(int, args.input_size.split(','))
input_size = (d, h, w)
print(input_size)
PancreasDataSet(args.train_list, max_iters=args.num_steps * args.batch_size, crop_size=input_size,
                        scale=args.random_scale, mirror=args.random_mirror)

PancreasValDataSet(args.val_list)
# BraTSDataSet(args.data_dir, args.train_list, max_iters=args.num_steps * args.batch_size, crop_size=input_size,
#                         scale=args.random_scale, mirror=args.random_mirror)
# (self, root, list_path, max_iters=None, crop_size=(128, 160, 200), scale=True, mirror=True, ignore_label=255):