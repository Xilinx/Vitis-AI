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
import numpy as np
from tqdm import tqdm
from collections import OrderedDict
import torch
from torch.utils import data
import torchvision.transforms as transform
import torch.nn.functional as F
from PIL import Image
import argparse
import logging
import glob
from code.configs.model_config import Options

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
    test_data = data.DataLoader(testset, batch_size=args.test_batch_size,drop_last=False, shuffle=False)
    return test_data

def build_model(args):
    from code.models import fpn
    model = fpn.get_fpn(nclass=args.num_classes, backbone=args.backbone, pretrained=False)
    mode = model.cuda()
    checkpoint = torch.load(args.weight, map_location='cuda:0')
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

def visulization(args, model):
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
            scale_image = scale_image.unsqueeze(0).to(0)
            output = model(scale_image)
            if isinstance(output, (tuple, list)):
                output = output[0]
            output = output.cpu().data[0].numpy().transpose(1,2,0)
            seg_pred = np.asarray(np.argmax(output, axis=2), dtype=np.uint8)
            name = imgName.split('/')[-1]
            img_extn = imgName.split('.')[-1]
            color_mask = colorize_mask(seg_pred)
            color_mask.save(os.path.join(args.save_dir, name[:-4]+'_color.png'))


def eval_miou(data, model):
    from code.utils import miou_utils as utils
    confmat = utils.ConfusionMatrix(args.num_classes)
    tbar = tqdm(data, desc='\r')
    with torch.no_grad():
        for i, (image, target) in enumerate(tbar): 
            image, target = image.to(0), target.to(0)
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
    if not os.path.exists(args.save_dir):
       os.makedirs(args.save_dir)
    # model
    model = build_model(args)
    model.eval()
    if args.eval:
        print('===> Evaluation mIoU: ')
        test_data = build_data(args)
        eval_miou(test_data, model)    
    else:
        print('===> Visualization: ')
        visulization(args, model)

if __name__ == "__main__":
    args = Options().parse()
    torch.manual_seed(args.seed)
    main(args)
