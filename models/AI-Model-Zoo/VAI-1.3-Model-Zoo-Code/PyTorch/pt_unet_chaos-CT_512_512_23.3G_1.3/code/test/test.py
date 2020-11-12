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

import os,sys
if os.environ["W_QUANT"]=='1':
    # load quant apis
    from pytorch_nndct.apis import torch_quantizer, dump_xmodel

import logging
import warnings
import argparse
warnings.filterwarnings("ignore")
import torch
import torch.nn as nn
from torch.utils import data
import numpy as np
from tqdm import tqdm

sys.path.insert(0,'..')
from dataset.datasets import ChaosCTDataSet as DataSet
from networks.evaluate import evaluate_main
from networks.UNet import unet

IMG_MEAN = np.array((104.00698793,116.66876762,122.67891434), dtype=np.float32)

class Configs():
    def __init__(self):

        parser = argparse.ArgumentParser("2D-UNet-LW for Choas-CT segmentation")

        #dataset options
        parser.add_argument('--data_root', type=str, default='../data', help='path to dataset')

        parser.add_argument('--input_size', type=str, default='512,512', help='input size')
        parser.add_argument('--list_file', type=str, default='dataset/lists/val_img_seg.txt', help='validation list file')
        parser.add_argument('--weight', type=str, default=None, help='resume from weight')
        parser.add_argument('--classes_num', type=int, default=2, help='dataset class number')
        parser.add_argument('--ignore_index', type=int, default=255, help='dataset ignore index')

        #gpu options
        parser.add_argument('--gpu', type=int, default=0, help='gpu id')

        #quantization options
        parser.add_argument('--quant_dir', type=str, default='quantize_result', help='path to save quant info')
        parser.add_argument('--quant_mode', default='calib', choices=['float', 'calib', 'test'], \
                                            help='quantization mode. 0: no quantization, evaluate float model, calib: quantize, test: evaluate quantized model')
        parser.add_argument('--finetune', dest='finetune', action='store_true', help='finetune model before calibration')
        parser.add_argument('--dump_xmodel', dest='dump_xmodel', action='store_true', help='dump xmodel after test')
        parser.add_argument('--device', default='gpu', choices=['gpu', 'cpu'], help='assign runtime device')

        self.parser = parser

    def parse(self):
        args = self.parser.parse_args()
        print(args)
        return args


class Criterion(nn.Module):
    '''
    DSN : We need to consider two supervision for the model.
    '''
    def __init__(self, ignore_index=255, weight=None, use_weight=True, reduce=True):
        super(Criterion, self).__init__()
        #class_wts = torch.ones(len(weight))
        self.ignore_index = ignore_index
        self.criterion = torch.nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_index, reduce=reduce)
        if not reduce:
            print("disabled the reduce.")

    def forward(self, preds, target):
        h, w = target.size(1), target.size(2)
        scale_pred = F.upsample(input=preds, size=(h, w), mode='bilinear', align_corners=True)
        loss = self.criterion(scale_pred, target)
        return loss


def main():
    args = Configs().parse()

    if args.dump_xmodel:
        args.device='cpu'

    if args.device=='cpu':
        device = torch.device("cpu")
    else:
        device = torch.device("cuda")
        
    model = unet(n_classes=2).to(device)
    if os.path.isfile(args.weight):
        weight_state_dict = torch.load(args.weight, map_location=device)
        model_state_dict = model.state_dict()
        overlap_dtate_dict = {k:v for k,v in weight_state_dict.items() if k in model_state_dict}
        model_state_dict.update(overlap_dtate_dict)
        model.load_state_dict(model_state_dict)
        print("====> load weights sucessfully!!!")
    else:
        print('can not find the checkpoint.')
        exit(-1)

    
    valloader, H, W = get_dataset(args, 'ChaosCT')

    input = torch.randn([1, 3, H, W])

    if args.quant_mode == 'float':
        quant_model = model
    else:
        ## new api
        ####################################################################################
        quantizer = torch_quantizer(args.quant_mode, model, (input), output_dir=args.quant_dir, device=device)

        quant_model = quantizer.quant_model

    criterion = Criterion(ignore_index=255, weight=None, use_weight=False, reduce=True)
    loss_fn = criterion.to(device)


    if args.quant_mode == 'calib' and args.finetune == True:
        ft_loader = valloader
        quantizer.finetune(evaluate_main, (quant_model, ft_loader, loss_fn))

    mean_IU, IU_array, Dice = evaluate_main(quant_model, valloader, 0, H, W, args.classes_num, args.ignore_index, True, device)
    print('[val with {}x{}] mean_IU:{:.6f}  Dice:{}'.format(H, W, mean_IU, Dice))

    # handle quantization result
    if args.quant_mode == 'calib':
        quantizer.export_quant_config()
    if args.quant_mode == 'test' and args.dump_xmodel:
        #deploy_check= True if args.dump_golden_data else False
        dump_xmodel(args.quant_dir, deploy_check=True)

def get_dataset(args, data_set):
    h, w = map(int, args.input_size.split(','))
    valloader = data.DataLoader(DataSet(root=args.data_root, list_path='./dataset/lists/val_img_seg.txt', \
                      crop_size=(h, w), mean=IMG_MEAN, scale=True, mirror=False, ignore_label=args.ignore_index), \
                      batch_size=1, shuffle=False, pin_memory=True)

    return valloader, h, w

if __name__ == '__main__':
    main()

