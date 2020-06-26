# Copyright 2019 Xilinx Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import logging
import os
import sys

import torch
from torch import nn

import network
from core.config import opt, update_config
from core.loader import get_data_provider
from core.solver import Solver
from ipdb import set_trace
from pytorch_nndct.apis import torch_quantizer, dump_xmodel

FORMAT = '[%(levelname)s]: %(message)s'
logging.basicConfig(
    level=logging.INFO,
    format=FORMAT,
    stream=sys.stdout
)


def test(args):
    logging.info('======= user config ======')
    logging.info(print(args))
    logging.info('======= end ======')

    train_data, test_data, num_query, num_class = get_data_provider(opt, args.dataset, args.dataset_root)

    net = getattr(network, opt.network.name)(num_class, 1)
    if args.load_model[-3:] == 'tar':
        checkpoint = torch.load(args.load_model)['state_dict']
        for i in checkpoint:
            if 'classifier' in i:
                continue
            net.state_dict()[i].copy_(checkpoint[i])
    else:
        #net.load_param(args.load_model)
        net.load_state_dict(torch.load(args.load_model))
    logging.info('load model checkpoint: {}'.format(args.load_model))
    
    net = nn.DataParallel(net).cuda()
    net = net.to(opt.device)
    
    x = torch.randn(1,3,256,128).cuda()
    quantizer = torch_quantizer(args.quant_mode, net.module, (x))
    net.module = quantizer.quant_model.to(opt.device)
    net.eval()
    mod = Solver(opt, net)
    mod.test_func(test_data, num_query)
     
    quantizer.export_quant_config()
    dump_xmodel('quantize_result')
   
def main():
    parser = argparse.ArgumentParser(description='reid model testing')
    parser.add_argument('--dataset', type=str, default = 'market1501', 
                        help = 'set the dataset for test')
    parser.add_argument('--dataset_root', type=str, default = '../../data/market1501',
                        help = 'dataset path')
    parser.add_argument('--config_file', type=str, default='./configs/market_softmax.yml',
                        help='Optional config file for params')
    parser.add_argument('--load_model', type=str, required=True, default = './log_models/market_softmax/model_best.pth.tar',
                        help='load trained model for testing')
    parser.add_argument('--device', type=str, default='gpu', choices=['gpu','cpu'],
                        help='set running device')
    parser.add_argument('--gpu', type=str, default='0',
                        help='set visible cuda device')
    parser.add_argument('--quant_mode', type=int, default=1)

    args = parser.parse_args()
    update_config(args.config_file)

    if args.device=='gpu':
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
        opt.device = torch.device('cuda:{}'.format(args.gpu))
    else:
        opt.device = torch.device('cpu')
    test(args)


if __name__ == '__main__':
    main()
