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

import random
import torch
import numpy as np
from torch.autograd import Variable
from torch.backends import cudnn
import torch.nn.functional as F

cudnn.benchmark = True
import torch.nn as nn
import logging
import argparse

class Configs():
    def __init__(self):
        parser = argparse.ArgumentParser("ENet(modified) on Cityscapes")
        #dataset options
        parser.add_argument('--dataset', type=str, default='cityscapes', help='dataset name')
        parser.add_argument('--data_root', type=str, default='./data/cityscapes', help='path to dataset')
        parser.add_argument('--num_classes', type=int, default=19, help='classes numbers')
        parser.add_argument('--ignore_label', type=int, default=255, help='ignore index')

        parser.add_argument('--checkpoint_dir', type=str, default='ckpt-cityscapes', help='path to checkpoint')
        parser.add_argument('--input_size', nargs='+', type=int, default=[1024, 512], help='input size')
        parser.add_argument('--weight', type=str, default=None, help='resume from weight')
        parser.add_argument('--test_only', action='store_true', help='if only test the trained model')
        parser.add_argument('--local_rank', type=int, default=0)
        parser.add_argument('--gpu_num', type=int, default=1, help='number of gpus')
        parser.add_argument('--print_freq', type=int, default=20, help='print frequency')
        #validation options
        parser.add_argument('--val_batch_size', type=int, default=1, help='batch size')
        # evaluation miou options
        parser.add_argument('--eval', action='store_true', help='evaluation miou mode')
        # demo options
        parser.add_argument('--demo_dir', type=str, default='./data/demo/', help='path to demo dataset')
        parser.add_argument('--save_dir', type=str, default='./data/demo_results', help='path to save demo prediction')

        parser.add_argument('--quant_dir', type=str, default='quantize_result', help='path to save quant info')
        parser.add_argument('--quant_mode', default='calib', choices=['float', 'calib', 'test'], \
                                            help='quantization mode. 0: no quantization, evaluate float model, calib: quantize, test: evaluate quantized model')
        parser.add_argument('--finetune', dest='finetune', action='store_true', help='finetune model before calibration')
        parser.add_argument('--dump_xmodel', dest='dump_xmodel', action='store_true', help='dump xmodel after test')
        parser.add_argument('--device', default='gpu', choices=['gpu', 'cpu'], help='assign runtime device')

        self.parser = parser

    def parse(self):
        args = self.parser.parse_args()
        return args

class Criterion(nn.Module):
    '''
    DSN : We need to consider two supervision for the model.
    '''
    def __init__(self, ignore_index=255, weight=None, use_weight=True, reduce=True):
        super(Criterion, self).__init__()
        self.ignore_index = ignore_index
        self.criterion = torch.nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_index, reduce=reduce)
        if not reduce:
            print("disabled the reduce.")

    def forward(self, preds, target):
        loss = self.criterion(pred, target)
        return loss


def main(args):

    if args.dump_xmodel:
        args.device='cpu'
        args.val_batch_size=1

    if args.device=='cpu':
        device = torch.device("cpu")
    else:
        device = torch.device("cuda")
    # network
    from code.models.enet_xilinx import ENet
    print('====> Bulid Networks...')
    net = ENet(args.num_classes).to(device)

    if os.path.isfile(args.weight):
        state_dict = torch.load(args.weight, map_location=device)  
        net.load_state_dict(state_dict['state_dict'])
        print("====> load weights sucessfully!!!")
    else:
        logging.error('can not find the checkpoint.')
        exit(-1)

    net.eval()

    input = torch.randn([1, 3, 512, 1024])

    criterion = Criterion(ignore_index=255, weight=None, use_weight=False, reduce=True)
    loss_fn = criterion.to(device)
    
    if args.quant_mode == 'float':
        quant_model = net
    else:
        ## new api
        ####################################################################################
        quantizer = torch_quantizer(args.quant_mode, net, (input), output_dir=args.quant_dir, device=device)

        quant_model = quantizer.quant_model

    if args.eval:
        print('====> Evaluation mIoU')
        eval_miou(args, quant_model, device)    

        # handle quantization result
        if args.quant_mode == 'calib':
            quantizer.export_quant_config()
        if args.quant_mode == 'test' and args.dump_xmodel:
            #deploy_check= True if args.dump_golden_data else False
            dump_xmodel(args.quant_dir, deploy_check=True)
    else:
        print('====> Prediction for demo')
        demo(args, quant_model, device)



def eval_miou(args, net, device):
    from code.utils import evaluate
    from code.data_loader import cityscapes as citys
    from torch.utils.data import DataLoader
    import torchvision.transforms as standard_transforms
    # data
    print('====> Bulid Dataset...')

    assert args.dataset in ['cityscapes', 'camvid', 'coco', 'voc']
    val_set = citys.CityScapes(root=args.data_root, quality='fine', mode='val', size=(1024, 512))
    val_loader = DataLoader(val_set, batch_size=args.val_batch_size, shuffle=False)

    inputs_all, gts_all, predictions_all = [], [], []
    with torch.no_grad():
        for vi, data in enumerate(val_loader):
            print('Process batch id: {} / {}'.format(vi, len(val_loader)))
            inputs, gts = data
            inputs = inputs.to(device)
            gts = gts.to(device)
            outputs = net(inputs)
            if outputs.size()[2:] != gts.size()[1:]:
                outputs = F.interpolate(outputs, size=gts.size()[1:], mode='bilinear', align_corners=True)           
            predictions = outputs.data.max(1)[1].squeeze_(1).cpu().numpy()
            inputs_all.append(inputs.data.cpu())
            gts_all.append(gts.data.cpu().numpy())
            predictions_all.append(predictions)
            if args.dump_xmodel:
                return
        gts_all = np.concatenate(gts_all)
        predictions_all = np.concatenate(predictions_all)
        acc, acc_cls, ious, mean_iu, fwavacc = evaluate(predictions_all, gts_all, args.num_classes, args.ignore_label)
        print('>>>>>>>>>>>>>>>Evaluation Results:>>>>>>>>>>>>>>>>>>>')
        print('Mean IoU(%): {}'.format(mean_iu * 100))
        print('Per-class Mean IoUs: {}'.format(ious))

def demo(args, net, device):
    import cv2, glob
    from PIL import Image
    if not os.path.isdir(args.save_dir):
        os.mkdir(args.save_dir)
    # read all the images in the folder
    image_list = glob.glob(args.demo_dir + '*')

    # color pallete
    pallete = [128, 64, 128, 244, 35, 232, 70, 70, 70, 102, 102, 156, 190, 153, 153, 153, 153, 153, 250, 170, 30,
               220, 220, 0, 107, 142, 35, 152, 251, 152, 70, 130, 180, 220, 20, 60, 255, 0, 0, 0, 0, 142, 0, 0, 70,
               0, 60, 100, 0, 80, 100, 0, 0, 230, 119, 11, 32 ]

    mean = [.485, .456, .406]
    std =  [.229, .224, .225]

    for i, imgName in enumerate(image_list):
        name = imgName.split('/')[-1]
        img = cv2.imread(imgName).astype(np.float32)
        H, W = img.shape[0], img.shape[1]
        # image normalize
        img = cv2.resize(img, (args.input_size[0], args.input_size[1]))
        img =  img / 255.0
        for j in range(3):
            img[:, :, j] -= mean[j]
        for j in range(3):
            img[:, :, j] /= std[j]
        img = img.transpose((2, 0, 1))
        img_tensor = torch.from_numpy(img)
        img_tensor = torch.unsqueeze(img_tensor, 0)  # add a batch dimension
        with torch.no_grad():
            img_variable = img_tensor.to(device)
            outputs = net(img_variable)
            if outputs.size()[-1] != W:
                outputs = F.interpolate(outputs, size=(H, W), mode='bilinear', align_corners=True)

            classMap_numpy = outputs[0].max(0)[1].byte().cpu().data.numpy()
            classMap_numpy = Image.fromarray(classMap_numpy)

            name = imgName.split('/')[-1]
            classMap_numpy_color = classMap_numpy.copy()
            classMap_numpy_color.putpalette(pallete)
            classMap_numpy_color.save(os.path.join(args.save_dir, 'color_' + name))

if __name__ == '__main__':
    args = Configs().parse()
    main(args)

