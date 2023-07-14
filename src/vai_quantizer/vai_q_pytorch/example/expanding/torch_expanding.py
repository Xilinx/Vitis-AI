from torch import Tensor, nn
import torch
import numpy as np
from pytorch_nndct.expanding.structured import ExpandingRunner
from torchvision.models.resnet import resnet18, resnet34, resnet50, resnet152
from torchvision.models.inception import inception_v3
import argparse

parser = argparse.ArgumentParser()

parser.add_argument(
    '--channel-divisible', type=int, required=False, default=4,
    help='Number of conv channels will be expanded to multiply of channel_divisible')

args, _ = parser.parse_known_args()

class MnistConvnet(nn.Module):
  def __init__(self) -> None:
    super().__init__()
    self.conv1 = nn.Conv2d(1, 32, 3)
    self.relu1 = nn.ReLU(True)
    self.bn1 = nn.BatchNorm2d(32)
    self.max_pooling1 = nn.MaxPool2d(2)
    self.conv2 = nn.Conv2d(32, 64, 2)
    self.relu2 = nn.ReLU(True)
    self.instance_norm = nn.InstanceNorm2d(64, affine=True)
    self.max_pooling2 = nn.MaxPool2d(2)
    self.flatten = nn.Flatten()
    self.linear = nn.Linear(2304, 10)

  def forward(self, x):
    x = self.conv1(x)
    x = self.relu1(x)
    x = self.bn1(x)
    x = self.max_pooling1(x)
    x = self.conv2(x)
    x = self.relu2(x)
    x = self.instance_norm(x)
    x = self.max_pooling2(x)
    x = self.flatten(x)
    x = self.linear(x)
    return x

def do_expanding(model: nn.Module, input_signature: Tensor, channel_divisible: int):
  expanding_runner = ExpandingRunner(model, input_signature)
  expanded_model, _ = expanding_runner.expand(channel_divisible)
  result1 = model.eval()(input_signature).detach().numpy()
  result2 = expanded_model.eval()(input_signature).detach().numpy()
  abs_diff = abs(result1 - result2)
  relative_diff = abs_diff / np.concatenate((abs(result1), abs(result2)), axis=0).max(axis=0)
  print("min_relative_diff = {}\nmax_relative_diff = {}\naverate_relative_diff = {}".\
    format(relative_diff.min(), relative_diff.max(), relative_diff.mean()))


def mnist_expanding():
  model = MnistConvnet()
  input_signature = torch.rand([1, 1, 28, 28])
  print("expanding mnist")
  do_expanding(model, input_signature, args.channel_divisible)


def resnet18_expanding():
  model = resnet18()
  input_signature = torch.randn([1, 3, 224, 224], dtype=torch.float32)
  print("expanding resnet18")
  do_expanding(model, input_signature, args.channel_divisible)


def resnet34_expanding():
  model = resnet34()
  input_signature = torch.randn([1, 3, 224, 224], dtype=torch.float32)
  print("expanding resnet18")
  do_expanding(model, input_signature, args.channel_divisible)


def resnet50_expanding():
  model = resnet50()
  input_signature = torch.randn([1, 3, 224, 224], dtype=torch.float32)
  print("expanding resnet18")
  do_expanding(model, input_signature, args.channel_divisible)


def resnet152_expanding():
  model = resnet152()
  input_signature = torch.randn([1, 3, 224, 224], dtype=torch.float32)
  print("expanding resnet18")
  do_expanding(model, input_signature, args.channel_divisible)


def inception_expanding():
  model = inception_v3(init_weights=True)
  input_signature = torch.randn([1, 3, 299, 299], dtype=torch.float32)
  print("expanding inception_v3")
  do_expanding(model, input_signature, args.channel_divisible)


if __name__ == "__main__":
  mnist_expanding()
  resnet18_expanding()
  resnet34_expanding()
  resnet50_expanding()
  resnet152_expanding()
  inception_expanding()
