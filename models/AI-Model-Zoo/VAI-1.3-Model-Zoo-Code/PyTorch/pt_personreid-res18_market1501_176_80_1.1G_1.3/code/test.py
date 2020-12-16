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

if os.environ["W_QUANT"]=='1':
    import pytorch_nndct
    from pytorch_nndct.apis import torch_quantizer, dump_xmodel

import torch
from torch import nn

import network
from core.config import opt, update_config
from core.loader import get_data_provider
from core.solver import Solver
from ipdb import set_trace

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

    net = getattr(network, opt.network.name)(opt.network.backbone, num_class, opt.network.last_stride)
    checkpoint = torch.load(args.load_model, map_location=opt.device)
    for i in checkpoint:
        if 'classifier' in i:
            continue
        net.state_dict()[i].copy_(checkpoint[i])
    logging.info('load model checkpoint: {}'.format(args.load_model))
    
    if args.device=='gpu' and args.quant_mode=='float':
        net = nn.DataParallel(net).to(opt.device)
    net = net.to(opt.device)

    resize_wh = opt.aug.resize_size
    x = torch.randn(1,3,resize_wh[0],resize_wh[1]).to(opt.device)
    if args.quant_mode == 'float':
        quant_model = net
    else:
        quantizer = torch_quantizer(args.quant_mode, net, (x), output_dir=args.output_path, device=opt.device)
        quant_model = quantizer.quant_model.to(opt.device)
    quant_model.eval()
    mod = Solver(opt, quant_model)
    mod.test_func(test_data, num_query)
     
    if args.quant_mode == 'calib':
        quantizer.export_quant_config()
    if args.quant_mode == 'test' and args.dump_xmodel:
        dump_xmodel(output_dir=args.output_path, deploy_check=True)
   
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
                        help='assign running device')
    parser.add_argument('--quant_mode', default='calib', choices=['float', 'calib', 'test'],
                        help='quantization mode. 0: no quantization, evaluate float model, calib: quantize, test: evaluate quantized model')    
    parser.add_argument('--dump_xmodel', dest='dump_xmodel', action='store_true',
                        help='dump xmodel after test')
    parser.add_argument('--batch_size',default=32, type=int)
    parser.add_argument('--output_path', default='quantize_result')

    args = parser.parse_args()
    update_config(args.config_file)
    if args.dump_xmodel:
        args.batch_size=1
        args.device='cpu'

    if args.device=='gpu':
        opt.device = torch.device('cuda')
    else:
        opt.device = torch.device('cpu')
    opt.test.batch_size = args.batch_size

    test(args)


if __name__ == '__main__':
    main()
