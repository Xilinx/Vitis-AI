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


from __future__ import print_function, absolute_import

from collections import defaultdict

import numpy as np
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import Dataset, Sampler, DataLoader

from .data_manager import init_dataset


def read_image(img_path):
    """Keep reading image until succeed.
    This can avoid IOError incurred by heavy IO process."""
    got_img = False
    while not got_img:
        try:
            img = Image.open(img_path).convert("RGB")
            got_img = True
        except IOError:
            print("IOError incurred when reading '{}'. Will redo. Don't worry. Just chill.".format(img_path))
            pass
    return img


class ImageData(Dataset):
    def __init__(self, dataset, transform):
        self.dataset = dataset
        self.transform = transform

    def __getitem__(self, item):
        img, pid, camid = self.dataset[item]
        img = read_image(img)
        if self.transform is not None:
            img = self.transform(img)
        return img, pid, camid

    def __len__(self):
        return len(self.dataset)


def get_data_provider(opt, dataset_name, dataset_root):
    test_batch_size = opt.test.batch_size
   
    # data augmenter
    random_mirror = opt.aug.get('random_mirror', False)
    random_erase = opt.aug.get('random_erase', False)
    pad = opt.aug.get('pad', False)
    random_crop = opt.aug.get('random_crop', False)

    h, w = opt.aug.resize_size
    test_aug = list()
    test_aug.append(T.Resize((h, w)))
    test_aug.append(T.ToTensor())
    test_aug.append(T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.225, 0.225, 0.225]))
    test_aug = T.Compose(test_aug)

    dataset = init_dataset(dataset_name, dataset_root)
    pids = [a[1] for a in dataset.train]
    num_class = np.max(pids) + 1
         
    #dataset = init_dataset(dataset_name)
    test_set = ImageData(dataset.query + dataset.gallery, test_aug)
    test_loader = DataLoader(test_set, batch_size=test_batch_size, num_workers=opt.network.workers, pin_memory=True)
    return None, test_loader, len(dataset.query), num_class  # return number of query
