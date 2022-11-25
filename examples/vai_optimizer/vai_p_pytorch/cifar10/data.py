#!/usr/bin/env bash

# Copyright 2021 Xilinx Inc.
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

import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms

transform = transforms.Compose(
      [transforms.ToTensor(),
      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

def get_dataloader(data_dir, batch_size, num_workers=48, shuffle=True, train=True, download=True):
  dataset = datasets.CIFAR10(root=data_dir, train=train, download=download, transform=transform)
  data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
  return data_loader

def get_subnet_dataloader(data_dir, batch_size, subnet_len, num_workers=48, shuffle=True, train=True, download=True):
  dataset = datasets.CIFAR10(root=data_dir, train=train, download=download, transform=transform)
  subnet_dataset = torch.utils.data.Subset(dataset, list(range(subset_len)))
  data_loader = torch.utils.data.DataLoader(
      subnet_dataset,
      batch_size=batch_size,
      shuffle=shuffle,
      num_workers=num_workers,
      pin_memory=True)
  return data_loader
