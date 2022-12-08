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
import argparse
import torch
import wego_torch
import torchvision
import validators
import requests
from PIL import Image

import torchvision.transforms as transforms

from config import load_config

parser = argparse.ArgumentParser()

parser.add_argument('--img_url', default='', help='source input image.')
parser.add_argument('--float_model_path', default='', help='path of float model.')
parser.add_argument('--serialized_model_path', default='', help='path of serialize model.')
parser.add_argument('--calibration_images_folder', default='', help='folder which contains calibration images.')
parser.add_argument('--calib_set_len', default=200, type=int, help="number of images to use for PTQ calibration.")
parser.add_argument('--config_file', default='', help='model config file.')
parser.add_argument('--mode', default='normal', help="running mode.")
parser.add_argument('--serialize', action='store_true', default=False, help='serialize the compiled Model')

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

def get_wego_mod(img_transforms):

    input_shape = model_config['model']['input_shape']

    target_info = wego_torch.get_target_info()
    print("[Info] target_info: %s" % str(target_info))
    target_batch = target_info.batch

    if os.path.isdir(args.calibration_images_folder):
        dataset = torchvision.datasets.ImageFolder(args.calibration_images_folder, img_transforms)
    else:
        raise ValueError(f"calibration_images_folder doesn't exist: {args.calibration_images_folder}")

    # create a subset of dataset for calibration
    calib_dataset = torch.utils.data.Subset(dataset, list(range(args.calib_set_len)))

    # create dataloader for calibration dataset
    calib_dataloader = torch.utils.data.DataLoader(calib_dataset, batch_size=target_batch, shuffle=False)
    print(f"calibrate model with {len(calib_dataloader)} batches, batch size = {target_batch}")
    # define calibrator
    def calibrator(model, batch_data, batch_index, device):
        input_data, _ = batch_data
        input_data = input_data.to(device)
        model(input_data)

    # load original float model
    float_model = torchvision.models.inception_v3()
    if os.path.isfile(args.float_model_path):
        # https://download.pytorch.org/models/inception_v3_google-0cc3c7bd.pth
        float_model.load_state_dict(torch.load(args.float_model_path))
        print(f"successfully loaded float model from {args.float_model_path}")
    else:
        raise ValueError(f"float model file doesn't exist: {args.float_model_path}")

    # quantize
    quantized_model = wego_torch.quantize(float_model, [[target_batch] + input_shape], calib_dataloader, calibrator)

    # We can remove redudant fixneurons if we want to get better performance but the accuracy may has 
    # a minor difference with the original quantized model.
    if args.mode == 'normal':
        print("[Info] running in normal mode...")
        accuracy_mode = wego_torch.AccuracyMode.ReserveReduantFixNeurons
    elif args.mode == 'perf':
        print("[Info] running in perf mode...")
        accuracy_mode = wego_torch.AccuracyMode.Default
    else:
        raise ValueError('unsupport running mode - %s, support list: [normal, perf]' % (args.mode))

    # Create wego module by compiling the original qunatized torchscript model
    wego_mod = wego_torch.compile(quantized_model, wego_torch.CompileOptions(
        inputs_meta = [
        wego_torch.InputMeta(torch.float, [1] + input_shape)
        ],
        accuracy_mode=accuracy_mode
    ))

    return wego_mod

def main():
    print("running model:", model_config['model']['name'])

    # get preprocessing image transforms
    img_transforms = get_transform()

    # depending on the parameters passed
    # we could either load a previously compiled and serialized wego module
    # or quantize a float module and compile, serialize it
    if args.serialized_model_path:
        # deserialize and run the wego mod
        if os.path.isfile(args.serialized_model_path):
            wego_mod = torch.jit.load(args.serialized_model_path)
            print(f"successfully loaded model from {args.serialized_model_path}")
        else:
            raise ValueError(f"wego model file doesn't exist: {args.serialized_model_path}")
    elif args.float_model_path:
        # quantize the float model and compile it with wego api
        wego_mod = get_wego_mod(img_transforms)

        # you can serialize the compiled module to a file
        if args.serialize:
            print(f"serializing compiled wego module to inception_v3_wego_compiled.wg")
            wego_mod.save("inception_v3_wego_compiled.wg")
            print(f"serialization complete")

    # read image
    img = get_image_from_url(img_transforms, args.img_url)

    # run inference with image
    run_normal(img, wego_mod)

if __name__ == '__main__':
  main()
