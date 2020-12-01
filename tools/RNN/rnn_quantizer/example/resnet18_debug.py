import os
import re
import sys
import argparse
import time
import pdb
import datetime
import numpy as np
import random
import arg_parser
from tqdm import tqdm

from pytorch_nndct.apis import torch_quantizer
import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.models.resnet import resnet18

def load_data(args, **kwargs):
    # prepare data
    valdir = args.data_dir + '/val'
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    size = 224
    resize = 256
    starttime = time.time()
    dataset = torchvision.datasets.ImageFolder(valdir,)
    endtime = time.time()
    print ("Read img time: ", endtime-starttime)
    dataset = torchvision.datasets.ImageFolder(valdir,
            transforms.Compose([
                    transforms.Resize(resize),
                    transforms.CenterCrop(size),
                    transforms.ToTensor(),
                    normalize,
            ]))
    endtime = time.time()
    print ("Preprocess img time: ", endtime-starttime)

    assert args.subset_len <= len(dataset)
    if args.sample_method == 'random':
        dataset = torch.utils.data.Subset(
            dataset, random.sample(range(0, len(dataset)), args.subset_len))
    else:
        dataset = torch.utils.data.Subset(dataset, list(range(args.subset_len)))
    data_loader = torch.utils.data.DataLoader(
            dataset, args.batch_size, shuffle=False, **kwargs)
    return data_loader

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions
        for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def evaluate(device, model, val_loader):
    loss_fn = torch.nn.CrossEntropyLoss().to(device)
    model.eval()
    model = model.to(device)
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    total = 0
    Loss = 0
    # to get loss value after evaluation
    for iteraction, (images, labels) in tqdm(
            enumerate(val_loader), total=len(val_loader)):
        images = images.to(device)
        labels = labels.to(device)
        #pdb.set_trace()
        outputs = model(images)
        loss = loss_fn(outputs, labels)
        Loss += loss.item()
        total += images.size(0)
        acc1, acc5 = accuracy(outputs, labels, topk=(1, 5))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))
    acc1_gen = top1.avg
    acc5_gen = top5.avg
    loss_gen = Loss / total
    # logging accuracy
    print('loss: %g' % (loss_gen))
    print('top-1 / top-5 accuracy: %g / %g' % (acc1_gen, acc5_gen))

    return acc1_gen, acc5_gen, loss_gen

def quant_mode_0(args, device, file_path=''):
    model = resnet18().cpu()
    model.load_state_dict(torch.load(file_path))
    model.name = args.model_name

    val_loader = load_data(args)

    #register_modification_hooks(model_gen, train=False)
    acc1_gen, acc5_gen, loss_gen = evaluate(device, model, val_loader)

def quant_mode_1(args, device, file_path=''):
    model = resnet18().cpu()
    model.load_state_dict(torch.load(file_path))
    model.name = args.model_name

    input = torch.randn([args.batch_size, 3, 224, 224])
    quantizer = torch_quantizer(args.quant_mode, model, (input), device=device)
    val_loader = load_data(args)

    #register_modification_hooks(model_gen, train=False)
    acc1_gen, acc5_gen, loss_gen = evaluate(device, quantizer.quant_model, val_loader)

    quantizer.export_quant_config()

def quant_mode_2(args, device, file_path=''):
    model = resnet18().cpu()
    model.load_state_dict(torch.load(file_path))
    model.name = args.model_name

    input = torch.randn([args.batch_size, 3, 224, 224])
    quantizer = torch_quantizer(args.quant_mode, model, (input), device=device)
    val_loader = load_data(args)

    #register_modification_hooks(model_gen, train=False)
    acc1_gen, acc5_gen, loss_gen = evaluate(device, quantizer.quant_model, val_loader)

    # handle quantization result
    quantizer.export_quant_config()
    quantizer.export_xmodel(deploy_check=True)

if __name__ == '__main__':
    parser = arg_parser.get_resnet18_quant_parser()
    args, _ = parser.parse_known_args()
    print("args = ", args)

    # calibration or evaluation
    device = torch.device(args.device)
    file_path = os.path.join(args.model_dir, args.model_name + '.pth')
    if args.quant_mode == 0:
        quant_mode_0(args, device, file_path)
    elif args.quant_mode == 1:
        quant_mode_1(args, device, file_path)
    elif args.quant_mode == 2:
        quant_mode_2(args, device, file_path)
    else:
        print("Not support quant_mode ", args.quant_mode)
        sys.exit(1)

    print("-------- End of {} test ".format(args.model_name))

