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


import os
import sys
if os.environ["W_QUANT"]=='1':
    # load quant apis
    from pytorch_nndct.apis import torch_quantizer, dump_xmodel

import numpy as np
from tqdm import tqdm
from collections import OrderedDict
import torch
import torch.nn as nn
from torch.utils import data
import torchvision.transforms as transform
import torch.nn.functional as F
from PIL import Image
import argparse
import logging
import glob
#from code.configs.model_config import Options

class Configs():
    def __init__(self):
        parser = argparse.ArgumentParser(description='PyTorch SemanticFPN model')
        # model and dataset 
        parser.add_argument('--model', type=str, default='fpn', help='model name (default: fpn)')
        parser.add_argument('--backbone', type=str, default='resnet18',choices=['resnet18', 'mobilenetv2'], \
                             help='backbone name (default: resnet18)')
        parser.add_argument('--dataset', type=str, default='citys',help='dataset name (default: cityscapes)')
        parser.add_argument('--num-classes', type=int, default=19, help='the classes numbers (default: 19 for cityscapes)')
        parser.add_argument('--data-folder', type=str, default='./data/cityscapes',help='training dataset folder (default: ./data)')
        parser.add_argument('--ignore_label', type=int, default=-1, help='the ignore label (default: 255 for cityscapes)')

        parser.add_argument('--base-size', type=int, default=1024, help='the shortest image size')
        parser.add_argument('--crop-size', type=int, default=512, help='input size for inference')
        parser.add_argument('--batch-size', type=int, default=10,metavar='N', help='input batch size for testing (default: 10)')
        # cuda, seed and logging
        parser.add_argument('--workers', type=int, default=16, metavar='N', help='dataloader threads')
        parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')
        parser.add_argument('--seed', type=int, default=1, metavar='S',help='random seed (default: 1)')
        # checking point
        parser.add_argument('--weight', type=str, default=None, help='path to final weight')
        # evaluation option
        parser.add_argument('--eval', action='store_true', default=False, help='evaluating mIoU')
        # test option
        parser.add_argument('--scale', type=float, default=0.5, help='downsample scale')
        parser.add_argument('--test-folder', type=str, default=None, help='path to demo folder')
        parser.add_argument('--save-dir', type=str, default='./data/demo_results')

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
        args.cuda = not args.no_cuda and torch.cuda.is_available()
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


        
def build_data(args):
    from code.datasets import get_segmentation_dataset
    input_transform = transform.Compose([
        transform.ToTensor(),
        transform.Normalize([.485, .456, .406], [.229, .224, .225])])

    data_kwargs = {'transform': input_transform, 'base_size': args.base_size,'crop_size': args.crop_size}
    if args.eval:
        testset = get_segmentation_dataset(args.dataset, split='val', mode='testval', root=args.data_folder,**data_kwargs)
    # dataloader
    loader_kwargs = {'num_workers': args.workers, 'pin_memory': True} if args.cuda else {}
    test_data = data.DataLoader(testset, batch_size=args.batch_size,drop_last=False, shuffle=False)
    return test_data

def build_model(args, device):
    from code.models import fpn
    model = fpn.get_fpn(nclass=args.num_classes, backbone=args.backbone, pretrained=False).to(device)
    checkpoint = torch.load(args.weight, map_location=device)
    checkpoint['state_dict'] = OrderedDict([(k[5:], v) if 'base' in k else (k, v) for k, v in checkpoint['state_dict'].items()])
    model.load_state_dict(checkpoint['state_dict'])
    return model

def colorize_mask(mask):

    palette = [128, 64, 128, 244, 35, 232, 70, 70, 70, 102, 102, 156, 190, 153, 153, 153, 153, 153, 250, 170, 30,
               220, 220, 0, 107, 142, 35, 152, 251, 152, 70, 130, 180, 220, 20, 60, 255, 0, 0, 0, 0, 142, 0, 0, 70,
               0, 60, 100, 0, 80, 100, 0, 0, 230, 119, 11, 32]

    zero_pad = 256 * 3 - len(palette)
    for i in range(zero_pad):
        palette.append(0)
    new_mask = Image.fromarray(mask.astype(np.uint8)).convert('P')
    new_mask.putpalette(palette)
    return new_mask

def data_transform(img, im_size, mean, std):
    from torchvision.transforms import functional as FT
    img = img.resize(im_size, Image.BILINEAR)
    tensor = FT.to_tensor(img)  # convert to tensor (values between 0 and 1)
    tensor = FT.normalize(tensor, mean, std)  # normalize the tensor
    return tensor

def visulization(args, model, device):
    # output folder
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    # data transforms
    MEAN = (.485, .456, .406)
    STD = (.229, .224, .225)
   
    image_list = glob.glob(os.path.join(args.test_folder, "*"))
    header = 'Demo'
    with torch.no_grad():
        for i, imgName in tqdm(enumerate(image_list)):
            img = Image.open(imgName).convert('RGB')
            w, h = img.size
            tw, th = int(w*args.scale), int(h*args.scale) 
            scale_image = data_transform(img,(tw, th) , MEAN, STD)
            scale_image = scale_image.unsqueeze(0).to(device)
            output = model(scale_image)
            if isinstance(output, (tuple, list)):
                output = output[0]
            output = output.cpu().data[0].numpy().transpose(1,2,0)
            seg_pred = np.asarray(np.argmax(output, axis=2), dtype=np.uint8)
            name = imgName.split('/')[-1]
            img_extn = imgName.split('.')[-1]
            color_mask = colorize_mask(seg_pred)
            color_mask.save(os.path.join(args.save_dir, name[:-4]+'_color.png'))


def eval_miou(data, model, device):
    from code.utils import miou_utils as utils
    confmat = utils.ConfusionMatrix(args.num_classes)
    tbar = tqdm(data, desc='\r')
    with torch.no_grad():
        for i, (image, target) in enumerate(tbar): 
            image, target = image.to(device), target.to(device)
            output = model(image)
            if isinstance(output, (tuple, list)):
                output = output[0]
            if output.size()[2:] != target.size()[1:]:
                output = F.interpolate(output, size=target.size()[1:], mode='bilinear', align_corners=True)
            
            confmat.update(target.flatten(), output.argmax(1).flatten())

        confmat.reduce_from_all_processes()
    print('Evaluation Metric: ')
    print(confmat)

def main(args):
    if args.dump_xmodel:
        args.device='cpu'
        args.batch_size=1

    if args.device=='cpu':
        device = torch.device("cpu")
    else:
        device = torch.device("cuda")
    # model
    model = build_model(args, device)
    model.eval()
    #model.to(device)

    H, W = args.crop_size, 2*args.crop_size
    input = torch.randn([1, 3, H, W])

    if args.quant_mode == 'float':
        quant_model = model
    else:
        ## new api
        ####################################################################################
        quantizer = torch_quantizer(args.quant_mode, model, (input), output_dir=args.quant_dir,device=device)

        quant_model = quantizer.quant_model

    criterion = Criterion(ignore_index=255, weight=None, use_weight=False, reduce=True)
    loss_fn = criterion.to(device)

    if args.quant_mode == 'calib' and args.finetune == True:
        ft_loader = build_data(args)
        quantizer.finetune(eval_miou, (quant_model, ft_loader, loss_fn))

    if args.eval:
        print('===> Evaluation mIoU: ')
        test_data = build_data(args)
        eval_miou(test_data, quant_model, device)    
    else:
        print('===> Visualization: ')
        visulization(args, quant_model, device)

    # handle quantization result
    if args.quant_mode == 'calib':
        quantizer.export_quant_config()
    if args.quant_mode == 'test' and args.dump_xmodel:
        #deploy_check= True if args.dump_golden_data else False
        dump_xmodel(args.quant_dir, deploy_check=True)


if __name__ == "__main__":
    args = Configs().parse()
    torch.manual_seed(args.seed)
    main(args)
