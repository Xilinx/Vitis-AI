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
import collections
import argparse
import validators
import requests
from tqdm import tqdm
from PIL import Image
import torch
from pytorch_nndct.apis import torch_quantizer
import wego_torch
import torchvision

import torchvision.transforms as transforms

from config import load_config

parser = argparse.ArgumentParser()

parser.add_argument('--img_url', default='', help='source input image.')
parser.add_argument('--float_model_path', default='', help='path of float model.')
parser.add_argument('--quant_result_path', default='', help='path of the quantization result.')
parser.add_argument('--output_dir', default='dynamo', help='path of the quantization output dir')
parser.add_argument('--calibration_images_folder', default='', help='folder which contains calibration images.')
parser.add_argument('--calib_set_len', default=200, type=int, help="number of images to use for PTQ calibration.")
parser.add_argument('--config_file', default='', help='model config file.')
parser.add_argument('--phase', default='quantize', help="phase: 'quantize' for model quantization, 'compile' for model compilation and run with WeGO.")

args, _ = parser.parse_known_args()

model_config = load_config(args.config_file)

def get_image_from_url(img_transforms, url=''):
    # download the image from web if uri is valid.
    if (validators.url(url)):
        img = Image.open(requests.get(url, stream=True).raw)
    else:
        img = Image.open(url)

    img = img_transforms(img)
    return img

def get_categories():
    # https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt
    with open("imagenet_classes.txt", "r") as f:
        categories = [s.strip() for s in f.readlines()]
    return categories

def cal_topk(output, topk):
    topk_prob, topk_catid = torch.topk(output, topk)
    return topk_prob, topk_catid

def run_normal(img, wego_mod):
    with torch.no_grad():
        img = torch.unsqueeze(img, 0)
        output = wego_mod(img)
        prob = torch.nn.functional.softmax(output[0], dim=0)
        top5_prob, top5_catid = cal_topk(prob, 5)
        categories = get_categories()
        print("=====================TopK Result=====================")
        for i in range(top5_prob.size(0)):
            print(categories[top5_catid[i]], top5_prob[i].item())

def get_transform():
    mean = model_config['preprocess']['mean']
    std = model_config['preprocess']['std']
    size = model_config['preprocess']['input_size']
    resize = model_config['preprocess']['resize']
    print("[Info] preprocess mean - ", mean, ", std - ", std)

    img_transforms = transforms.Compose([
        transforms.Resize(resize),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    return img_transforms

def wego_dynamo_quantize(float_model, input_shape, calib_dataloader, device, output_dir):

    def calibrator(quant_model, dataloader, device):
        # some dataloaders don't support len
        if isinstance(dataloader, collections.abc.Sized):
            progress_bar = tqdm(enumerate(dataloader), total=len(dataloader))
        else:
            progress_bar = tqdm(enumerate(dataloader))

        # run calibration loop to get quant config
        with torch.no_grad():
            for b_idx, batch in progress_bar:
                input_data, _ = batch
                input_data = input_data.to(device)
                quant_model(input_data)

    # quantize the model and enable the dynamo flow by setting 'dynamo' with True.
    quantizer = torch_quantizer("calib", float_model, input_shape, device=device, output_dir=output_dir, dynamo = True)

    quant_model = quantizer.quant_model
    print(f"Calibration begin...")
    calibrator(quant_model, calib_dataloader, device)
    print(f"Caliration done, export quant configuration files to {output_dir}.")
    quantizer.export_quant_config()

def wego_dynamo_compile(float_model, quant_result_path):
    # We can remove redudant fixneurons if we want to get better performance but the accuracy may has 
    # a minor difference with the original quantized model.
    accuracy_mode = wego_torch.AccuracyMode.ReserveReduantFixNeurons
   
    # To create wego module by compiling the float model with dynamo mode enabled:
    # - Enable dynamo mode by setting 'dynamo' option with True
    # - Provide the original float model instead of the quantized torchscript model as input model
    # - Dynamo options must be provided with 'quantize_result_path' pointting to the quantized result path.
    # Note: when enabling dynamo mode, not input shape and other meta data are required to provided.
    dynamo_options = wego_torch.DynamoCompileOptions(
        quantize_result_path = quant_result_path
    )
    wego_mod = wego_torch.compile(float_model, wego_torch.CompileOptions(
        dynamo = True,
        dynamo_options = dynamo_options,
        accuracy_mode=accuracy_mode
    ))

    return wego_mod

def main():
    print("running model:", model_config['model']['name'])

    # load original float model
    float_model = torchvision.models.inception_v3()
    float_model.cpu()
    float_model.eval()
    if os.path.isfile(args.float_model_path):
        # https://download.pytorch.org/models/inception_v3_google-0cc3c7bd.pth
        float_model.load_state_dict(torch.load(args.float_model_path))
        print(f"successfully loaded float model from {args.float_model_path}")
    else:
        raise ValueError(f"float model file doesn't exist: {args.float_model_path}")
    
    img_transforms = get_transform()
    device = torch.device("cpu")

    if args.phase == "quantize":
        print("running in quantize mode.")

        target_info = wego_torch.get_target_info()
        print("[Info] target_info: %s" % str(target_info))
        target_batch = target_info.batch

        # get input shape of the model
        input_shape = model_config['model']['input_shape']
        input_shape = [[target_batch] + input_shape]

        # get preprocessing image transforms and create a dataloder
        if os.path.isdir(args.calibration_images_folder):
            dataset = torchvision.datasets.ImageFolder(args.calibration_images_folder, img_transforms)
        else:
            raise ValueError(f"calibration_images_folder doesn't exist: {args.calibration_images_folder}")

        # create a subset of dataset for calibration
        calib_dataset = torch.utils.data.Subset(dataset, list(range(args.calib_set_len)))

        # create dataloader for calibration dataset
        calib_dataloader = torch.utils.data.DataLoader(calib_dataset, batch_size=target_batch, shuffle=False)
        print(f"calibrate model with {len(calib_dataloader)} batches, batch size = {target_batch}")
        
        # run the quantizer in dynamo mode.
        wego_dynamo_quantize(float_model, input_shape, calib_dataloader, device, args.output_dir)

    elif args.phase == "compile":
        print("running in compile mode with dynamo enabled!")
        # compile the float in dynamo mode with WeGO
        wego_mod = wego_dynamo_compile(float_model, args.quant_result_path)
        # run inference with real image 
        img = get_image_from_url(img_transforms, args.img_url)
        run_normal(img, wego_mod)
    else:
        raise ValueError(f"Unknow phase ${args.phase} value, support list: [quantize, compile].")

if __name__ == '__main__':
  main()