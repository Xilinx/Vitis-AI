
import argparse
import numpy as np
from lstm_compiler.module import LstmModuleCompiler
from xir import Graph
from pathlib import Path

xmodel_path = '/scratch/workspace/wangke/dataset/compiler-data/openie/xmodel/'

def parse_args():
    parser = argparse.ArgumentParser(description='Lstm compiler test')
    parser.add_argument('--model_path',
                        type=str, 
                        default=xmodel_path, 
                        help='X model path')
    parser.add_argument('--device',
                        type=str,
                        default='u50',
                        help='Device to deploy the model')
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    print(args)
    compiler = LstmModuleCompiler(xmodels=args.model_path, device=args.device)
    compiler.compile()
    
if __name__ == '__main__':
    main()
    print('Success')
