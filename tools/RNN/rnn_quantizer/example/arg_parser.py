import os
import re
import sys
import argparse
import time
import pdb
import datetime
import numpy as np
import random

def get_resnet18_quant_parser():
    parser = argparse.ArgumentParser() 
    parser.add_argument(
        '--model_name',
        type=str,
        default="resnet18")
    parser.add_argument(
        '--data_dir',
        type=str,
        default="/proj/rdi/staff/niuxj/imagenet/",
        help='Data set directory, when quant_mode=1, it is for calibration, while quant_mode=2 it is for evaluation')
    parser.add_argument(
        '--model_dir',
        type=str,
        default="/proj/rdi/staff/wluo/UNIT_TEST/models",
        help='Trained model file path. Download pretrained model from the following url and put it in model_dir specified path: https://download.pytorch.org/models/resnet18-5c106cde.pth')
    parser.add_argument(
        '--subset_len',
        type=int,
        default=50000)
    parser.add_argument(
        '--sample_method',
        type=str,
        default='order',
        choices=['random', 'order'])
    parser.add_argument(
        '--batch_size',
        type=int,
        default=32)
    parser.add_argument(
        '--quant_mode', 
        type=int,
        default=1)
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        choices=['cpu', 'cuda'])
    return parser

# extension
def get_xxx_parser():
    parser = argparse.ArgumentParser() 
    parser.add_argument(
        '--model_name',
        type=str,
        default="xxx")
    return parser

