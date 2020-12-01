
import os
import argparse
import json
from lstm_simulator import LstmModuleSimulator

data_path = '/scratch/workspace/wangke/dataset/compiler-data/openie/'

def parse_args():
    parser = argparse.ArgumentParser(description='Lstm simulator test')
    parser.add_argument('--data_path',
                        type=str, 
                        default=data_path, 
                        help='Compilor data path')
    parser.add_argument('--data_check',
                        action='store_true',
                        help='Data check or not')
    args = parser.parse_args()
    return args
        
def main():
    args = parse_args()
    print(args)
    module_simulator = LstmModuleSimulator(data_path=args.data_path)
    module_simulator.run(data_check=args.data_check)
    
if __name__ == '__main__':
    main()
    print('Success')
