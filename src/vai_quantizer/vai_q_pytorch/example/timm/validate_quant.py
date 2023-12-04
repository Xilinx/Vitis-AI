# Copyright 2019 Ross Wightman

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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


import argparse
import json
import logging
import time
from collections import OrderedDict
from contextlib import suppress

import torch
import torch.nn as nn
import torch.nn.parallel

from timm.data import create_dataset, create_loader, resolve_data_config, RealLabelsImagenet
from timm.models import create_model, load_checkpoint
from timm.utils import accuracy, AverageMeter, setup_default_logging, ParseKwargs

_logger = logging.getLogger('validate')


parser = argparse.ArgumentParser(description='PyTorch ImageNet Validation')
parser.add_argument('data', nargs='?', metavar='DIR', const=None,
                    help='path to dataset (*deprecated*, use --data-dir)')
parser.add_argument('--data-dir', metavar='DIR',
                    help='path to dataset (root dir)')
parser.add_argument('--dataset', metavar='NAME', default='',
                    help='dataset type + name ("<type>/<name>") (default: ImageFolder or ImageTar if empty)')
parser.add_argument('--split', metavar='NAME', default='validation',
                    help='dataset split (default: validation)')
parser.add_argument('--dataset-download', action='store_true', default=False,
                    help='Allow download of dataset for torch/ and tfds/ datasets that support it.')
parser.add_argument('--model', '-m', metavar='NAME', default='dpn92',
                    help='model architecture (default: dpn92)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--img-size', default=None, type=int,
                    metavar='N', help='Input image dimension, uses model default if empty')
parser.add_argument('--use-train-size', action='store_true', default=False,
                    help='force use of train input size, even when test size is specified in pretrained cfg')
parser.add_argument('--crop-pct', default=None, type=float,
                    metavar='N', help='Input image center crop pct')
parser.add_argument('--crop-mode', default=None, type=str,
                    metavar='N', help='Input image crop mode (squash, border, center). Model default if None.')
parser.add_argument('--mean', type=float, nargs='+', default=None, metavar='MEAN',
                    help='Override mean pixel value of dataset')
parser.add_argument('--std', type=float,  nargs='+', default=None, metavar='STD',
                    help='Override std deviation of of dataset')
parser.add_argument('--interpolation', default='', type=str, metavar='NAME',
                    help='Image resize interpolation type (overrides model)')
parser.add_argument('--num-classes', type=int, default=None,
                    help='Number classes in dataset')
parser.add_argument('--class-map', default='', type=str, metavar='FILENAME',
                    help='path to class to idx mapping file (default: "")')
parser.add_argument('--gp', default=None, type=str, metavar='POOL',
                    help='Global pool type, one of (fast, avg, max, avgmax, avgmaxc). Model default if None.')
parser.add_argument('--log-freq', default=32, type=int,
                    metavar='N', help='batch logging frequency (default: 32)')
parser.add_argument('--checkpoint', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--test-pool', dest='test_pool', action='store_true',
                    help='enable test time pool')
parser.add_argument('--no-prefetcher', action='store_true', default=False,
                    help='disable fast prefetcher')
parser.add_argument('--pin-mem', action='store_true', default=False,
                    help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
parser.add_argument('--channels-last', action='store_true', default=False,
                    help='Use channels_last memory layout')
parser.add_argument('--device', default='cuda', type=str,
                    help="Device (accelerator) to use.")
parser.add_argument('--use-ema', dest='use_ema', action='store_true',
                    help='use ema version of weights if present')
parser.add_argument('--model-kwargs', nargs='*', default={}, action=ParseKwargs)


scripting_group = parser.add_mutually_exclusive_group()
scripting_group.add_argument('--torchscript', default=False, action='store_true',
                             help='torch.jit.script the full model')

parser.add_argument('--real-labels', default='', type=str, metavar='FILENAME',
                    help='Real labels JSON file for imagenet evaluation')
parser.add_argument('--valid-labels', default='', type=str, metavar='FILENAME',
                    help='Valid label indices txt file for validation of partial label space')

# Quantization Args
parser.add_argument('--quantized_out', metavar='NAME', default='quantize_result',
                    help='quantized_out path')
parser.add_argument('--quant_mode', default='float',type=str,help='Should be one of float/calib/test')
parser.add_argument('--config_file', default=None,type=str,help='Provide quant_config.json')
parser.add_argument('--deploy', default=False,action='store_true', help='Deploy model')
parser.add_argument('--fast_finetune', default=False,action='store_true', help='Fast finetune the model')
parser.add_argument('--inspect', default=False,action='store_true', help='Inspect model')
parser.add_argument( '--subset_len', default=None, type=int, help='Subset length to evaluate model, using the whole validation dataset if it is not set')
parser.add_argument( '--ff_subset_len', default=None, type=int, help='Subset length to evaluate model of fast finetune, using the whole validation dataset if it is not set')

def validate(args):
    # Setp 1: Prepare device, float model and data
    args.pretrained = args.pretrained or not args.checkpoint
    args.prefetcher = not args.no_prefetcher

    device = torch.device(args.device)

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
    elif args.device == 'cuda':
        device = torch.device('cpu')
    if args.deploy:
        device = torch.device("cpu")
    
    in_chans = 3

    model = create_model(
        args.model,
        pretrained=args.pretrained,
        num_classes=args.num_classes,
        in_chans=in_chans,
        global_pool=args.gp,
        scriptable=args.torchscript,
        **args.model_kwargs,
    )
    model.to(device)
    if args.num_classes is None:
        assert hasattr(model, 'num_classes'), 'Model must have `num_classes` attr if not set on cmd line/config.'
        args.num_classes = model.num_classes

    if args.checkpoint:
        load_checkpoint(model, args.checkpoint, args.use_ema)

    param_count = sum([m.numel() for m in model.parameters()])
    _logger.info('Model %s created, param count: %d' % (args.model, param_count))

    data_config = resolve_data_config(
        vars(args),
        model=model,
        use_test_size=not args.use_train_size,
        verbose=True,
    )
    _logger.info('batch size: %d' % args.batch_size)
    _logger.info('data config: %s' % data_config)
    test_time_pool = False

    criterion = nn.CrossEntropyLoss().to(device)

    root_dir = args.data or args.data_dir
    dataset = create_dataset(
        root=root_dir,
        name=args.dataset,
        split=args.split,
        download=args.dataset_download,
        class_map=args.class_map,
    )

    if args.valid_labels:
        with open(args.valid_labels, 'r') as f:
            valid_labels = [int(line.rstrip()) for line in f]
    else:
        valid_labels = None

    if args.real_labels:
        real_labels = RealLabelsImagenet(dataset.filenames(basename=True), real_json=args.real_labels)
    else:
        real_labels = None

    crop_pct = 1.0 if test_time_pool else data_config['crop_pct']
    _ = create_loader(
        dataset,
        input_size=data_config['input_size'],
        batch_size=args.batch_size,
        use_prefetcher=args.prefetcher,
        interpolation=data_config['interpolation'],
        mean=data_config['mean'],
        std=data_config['std'],
        num_workers=args.workers,
        crop_pct=crop_pct,
        crop_mode=data_config['crop_mode'],
        pin_memory=args.pin_mem,
        device=device
    )

    def set_data_loader(dataset, subset_len):
        sub_dataset = dataset
        if subset_len and subset_len > 0:
            sample_method = 'random'
            assert subset_len <= len(dataset)
            if sample_method == 'random':
                import random
                sub_dataset = torch.utils.data.Subset(
                    dataset, random.sample(range(0, len(dataset)), subset_len))
            else:
                sub_dataset = torch.utils.data.Subset(dataset, list(range(subset_len)))
        crop_pct = 1.0 if test_time_pool else data_config['crop_pct']
        sub_loader = create_loader(
            sub_dataset,
            input_size=data_config['input_size'],
            batch_size=args.batch_size,
            use_prefetcher=args.prefetcher,
            interpolation=data_config['interpolation'],
            mean=data_config['mean'],
            std=data_config['std'],
            num_workers=args.workers,
            crop_pct=crop_pct,
            crop_mode=data_config['crop_mode'],
            pin_memory=args.pin_mem,
            device=device
        )
        return sub_loader

    def simple_forward_loop(model, data_loader, device):
        model.eval()
        model = model.to(device)
        from tqdm import tqdm
        with torch.no_grad():
            for _, (input, _) in enumerate(tqdm(data_loader)):
                input = input.to(device)
                model(input)

    # Setp 2: Set quantization options for TIMM models
    from nndct_shared.utils import option_util
    option_util.set_option_value('nndct_leaky_relu_approximate', False)
    quant_out_dir = f"{args.quantized_out}/{args.model}" 

    # Setp 3: Inspect the model for specific target(Optional)
    if args.quant_mode in ["float"] and args.inspect:
        from pytorch_nndct.apis import Inspector
        dummy_input = torch.randn((args.batch_size,) + tuple(data_config['input_size'])).to(device)

        target = "DPUCAHX8L_ISA0_SP"
        inspector = Inspector(target)
        inspector.inspect(model, (dummy_input), device=device, output_dir=quant_out_dir + f"/{args.model}_inspect")
        import sys
        sys.exit(0)

    # Setp 4: Create a quantizer object and configure its settings. Quantization is not done here if it needs calibration.
    if args.quant_mode in ["calib", "test"]:
        from pytorch_nndct.apis import torch_quantizer
        dummy_input = torch.randn((args.batch_size,) + tuple(data_config['input_size'])).to(device)

        quantizer = torch_quantizer(args.quant_mode, model, input_args=(dummy_input), output_dir=quant_out_dir, device=device,quant_config_file=args.config_file)
        model = quantizer.quant_model.to(device)

    model.eval()
    
    # Setp 5: Fast Finetune(Optional)
    if args.fast_finetune == True:
        if args.quant_mode == 'calib':
            ff_data_loader = set_data_loader(dataset, args.ff_subset_len)
            quantizer.fast_finetune(simple_forward_loop, (model, ff_data_loader, device))
        elif args.quant_mode == 'test':
            quantizer.load_ft_param()
   
    # Setp 6: Evaluation
    # This function call is to do forward loop for model to be quantized.
    # Quantization calibration will be done after it if quant_mode is 'calib'.
    # Quantization test will be done after it if quant_mode is 'test'.
    eval_data_loader = set_data_loader(dataset, args.subset_len)
    if args.quant_mode == 'calib':
        simple_forward_loop(model, eval_data_loader, device)
    else:
        batch_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()
        with torch.no_grad():
            end = time.time()
            for batch_idx, (input, target) in enumerate(eval_data_loader):
                if args.no_prefetcher:
                    target = target.to(device)
                    input = input.to(device)
                if args.channels_last:
                    input = input.contiguous(memory_format=torch.channels_last)

                # compute output
                with suppress():
                    output = model(input)

                    if valid_labels is not None:
                        output = output[:, valid_labels]
                    loss = criterion(output, target)

                if real_labels is not None:
                    real_labels.add_result(output)

                # measure accuracy and record loss
                acc1, acc5 = accuracy(output.detach(), target, topk=(1, 5))
                losses.update(loss.item(), input.size(0))
                top1.update(acc1.item(), input.size(0))
                top5.update(acc5.item(), input.size(0))

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                if batch_idx % args.log_freq == 0:
                    _logger.info(
                        'Test: [{0:>4d}/{1}]  '
                        'Time: {batch_time.val:.3f}s ({batch_time.avg:.3f}s, {rate_avg:>7.2f}/s)  '
                        'Loss: {loss.val:>7.4f} ({loss.avg:>6.4f})  '
                        'Acc@1: {top1.val:>7.3f} ({top1.avg:>7.3f})  '
                        'Acc@5: {top5.val:>7.3f} ({top5.avg:>7.3f})'.format(
                            batch_idx,
                            len(eval_data_loader),
                            batch_time=batch_time,
                            rate_avg=input.size(0) / batch_time.avg,
                            loss=losses,
                            top1=top1,
                            top5=top5
                        )
                    )

    # Setp 7: Export quantization result
    if args.quant_mode == "calib":
        quantizer.export_quant_config()
        return

    # Setp 8: Deployment
    if args.quant_mode == "test" and args.deploy:
        quantizer.export_torch_script(output_dir=quant_out_dir)
        quantizer.export_xmodel(output_dir=quant_out_dir)
        quantizer.export_onnx_model(output_dir=quant_out_dir)
        return

    # Setp 9: Display inference result
    if real_labels is not None:
        # real labels mode replaces topk values at the end
        top1a, top5a = real_labels.get_accuracy(k=1), real_labels.get_accuracy(k=5)
    else:
        top1a, top5a = top1.avg, top5.avg
    results = OrderedDict(
        model=args.model,
        top1=round(top1a, 4), top1_err=round(100 - top1a, 4),
        top5=round(top5a, 4), top5_err=round(100 - top5a, 4),
        param_count=round(param_count / 1e6, 2),
        img_size=data_config['input_size'][-1],
        crop_pct=crop_pct,
        interpolation=data_config['interpolation'],
    )

    _logger.info(' * Acc@1 {:.3f} ({:.3f}) Acc@5 {:.3f} ({:.3f})'.format(
       results['top1'], results['top1_err'], results['top5'], results['top5_err']))

    print(f'--result\n{json.dumps(results, indent=4)}')

    return


def main():
    setup_default_logging()
    args = parser.parse_args()
    validate(args)


if __name__ == '__main__':
    main()