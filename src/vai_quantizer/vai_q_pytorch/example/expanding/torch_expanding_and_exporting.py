from pytorch_nndct.expanding.expanding_lib import expand_and_export, load_expanded_model
from torchvision.models.inception import inception_v3
import torch
from torch import nn
import os
import onnxruntime
import argparse
import numpy as np

parser = argparse.ArgumentParser()

parser.add_argument(
    '--channel-divisibles', type=str, required=False, default="4,8,16,32",
    help='specify a couple of channel_divisibles, join with ","')

parser.add_argument(
    '--out-dir', type=str, required=False, default="expanding_results",
    help='output directory path')

args, _ = parser.parse_known_args()

model_name = "inception_v3"
model = inception_v3(init_weights=True).eval()
input_signature = torch.rand((1, 3, 224, 224), dtype=torch.float32)
channel_divisibles = [int(i) for i in args.channel_divisibles.split(",")]
out_dir = args.out_dir


def do_expand_and_export():

  expand_and_export("inception_v3", model, input_signature, channel_divisibles, out_dir, 
    onnx_export_kwargs= {
      "input_names": ['input'],
      "output_names": ['output'],
      "dynamic_axes": {'input' : {0 : 'batch_size'},  
                        'output' : {0 : 'batch_size'}}
    })


def load_expanded_torch_model(model_name: str, model: nn.Module, input_signature: torch.Tensor, dir_path: str):
  expanding_spec_path = os.path.join(dir_path, "expanding_spec")
  model_path = os.path.join(dir_path, model_name + ".pth")
  return load_expanded_model(expanding_spec_path, model, input_signature, model_path)


def load_expanded_onnx_model(model_name: str, dir_path: str) -> onnxruntime.InferenceSession:
  model_path = os.path.join(dir_path, model_name + "_fp32.onnx")
  return onnxruntime.InferenceSession(model_path)


def summary(channel_divisible: int, raw_output: np.ndarray, torch_output: np.ndarray, onnx_output: np.ndarray) -> None:
  raw_torch_diff = abs(raw_output - torch_output)
  raw_torch_relative_diff = raw_torch_diff / np.concatenate((abs(raw_output), abs(torch_output)), axis=0).max(axis=0)
  raw_onnx_diff = abs(raw_output - onnx_output)
  raw_onnx_relative_diff = raw_onnx_diff / np.concatenate((abs(raw_output), abs(onnx_output)), axis=0).max(axis=0)
  print("relative diff for channel_divisible {}".format(channel_divisible))
  print("row-torch relative diff: max = {:.4f}, min = {:.4f}, average = {:.4f}"
    .format(raw_torch_relative_diff.max(), raw_torch_relative_diff.min(), raw_torch_relative_diff.mean()))
  print("row-onnx relative diff: max = {:.4f}, min = {:.4f}, average = {:.4f}"
    .format(raw_onnx_relative_diff.max(), raw_onnx_relative_diff.min(), raw_onnx_relative_diff.mean()))


def verify():
  for channel_divisible in channel_divisibles:
    dir_path = os.path.join(out_dir, model_name + "_padded_{}".format(channel_divisible))
    torch_model = load_expanded_torch_model(model_name, model, input_signature, dir_path).eval()
    onnx_model = load_expanded_onnx_model(model_name, dir_path)
    raw_output = model(input_signature).detach().numpy()
    torch_output = torch_model(input_signature).detach().numpy()
    onnx_output = onnx_model.run(None, {"input": input_signature.numpy()})[0]
    summary(channel_divisible, raw_output, torch_output, onnx_output)


if __name__ == "__main__":
  do_expand_and_export()
  verify()
