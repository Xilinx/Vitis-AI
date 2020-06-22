# (c) Copyright 2019 Xilinx, Inc. All rights reserved.
#
# This file contains confidential and proprietary information
# of Xilinx, Inc. and is protected under U.S. and
# international copyright and other intellectual property
# laws.
#
# DISCLAIMER
# This disclaimer is not a license and does not grant any
# rights to the materials distributed herewith. Except as
# otherwise provided in a valid license issued to you by
# Xilinx, and to the maximum extent permitted by applicable
# law: (1) THESE MATERIALS ARE MADE AVAILABLE "AS IS" AND
# WITH ALL FAULTS, AND XILINX HEREBY DISCLAIMS ALL WARRANTIES
# AND CONDITIONS, EXPRESS, IMPLIED, OR STATUTORY, INCLUDING
# BUT NOT LIMITED TO WARRANTIES OF MERCHANTABILITY, NON-
# INFRINGEMENT, OR FITNESS FOR ANY PARTICULAR PURPOSE; and
# (2) Xilinx shall not be liable (whether in contract or tort,
# including negligence, or under any other theory of
# liability) for any loss or damage of any kind or nature
# related to, arising under or in connection with these
# materials, including for any direct, or any indirect,
# special, incidental, or consequential loss or damage
# (including loss of data, profits, goodwill, or any type of
# loss or damage suffered as a result of any action brought
# by a third party) even if such damage or loss was
# reasonably foreseeable or Xilinx had been advised of the
# possibility of the same.
#
# CRITICAL APPLICATIONS
# Xilinx products are not designed or intended to be fail-
# safe, or for use in any application requiring fail-safe
# performance, such as life-support or safety devices or
# systems, Class III medical devices, nuclear facilities,
# applications related to the deployment of airbags, or any
# other applications that could lead to death, personal
# injury, or severe property or environmental damage
# (individually and collectively, "Critical
# Applications"). Customer assumes the sole risk and
# liability of any use of Xilinx products in Critical
# Applications, subject only to applicable laws and
# regulations governing limitations on product liability.
#
# THIS COPYRIGHT NOTICE AND DISCLAIMER MUST BE RETAINED AS
# PART OF THIS FILE AT ALL TIMES.

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
    logging.info('======= args ======')
    logging.info(print(args))
    logging.info('======= end ======')

    _, test_data, num_query, num_class = get_data_provider(opt, args.dataset, args.dataset_root)

    net = getattr(network, opt.network.name)(num_class, opt.network.last_stride)
    if args.load_model[-3:] == 'tar':
        checkpoint = torch.load(args.load_model)['state_dict']
        for i in checkpoint:
            if 'classifier' in i or 'fc' in i:
                continue
            net.state_dict()[i].copy_(checkpoint[i])
    else:
        net = net.load_state_dict(torch.load(args.load_model))
    logging.info('Load model checkpoint: {}'.format(args.load_model))
    
    if opt.device.type == 'cuda':
        net = nn.DataParallel(net)
    net = net.to(opt.device)
    x = torch.randn(1,3,96,96).cuda()
    quantizer = torch_quantizer(args.quant_mode, net.module, (x), output_dir = "quantize_result_large")
    net = quantizer.quant_model.cuda()
    net.eval()

    mod = Solver(opt, net)
    mod.test_func(test_data, num_query)

    quantizer.export_quant_config()
    dump_xmodel('quantize_result_large')
def main():
    parser = argparse.ArgumentParser(description='reid model testing')
    parser.add_argument('--dataset', type=str, default = 'facereid', 
                       help = 'set the dataset for test')
    parser.add_argument('--dataset_root', type=str, default = '../data/face_reid', 
                       help = 'dataset path')
    parser.add_argument('--config_file', type=str, required=True,
                        help='Optional config file for params')
    parser.add_argument('--load_model', type=str, required=True, 
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
