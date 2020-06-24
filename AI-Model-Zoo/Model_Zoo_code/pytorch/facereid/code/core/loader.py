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

from utils import augmenter
from .data_manager import init_dataset
from .random_erasing import RandomErasing
import copy
import random
from PIL import ImageFilter

from ipdb import set_trace

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


class RandomIdentitySampler(Sampler):
    """
    Randomly sample N identities, then for each identity,
    randomly sample K instances, therefore batch size is N*K.
    Args:
    - data_source (list): list of (img_path, pid, camid).
    - num_instances (int): number of instances per identity in a batch.
    - batch_size (int): number of examples in a batch.
    """

    def __init__(self, data_source, batch_size, num_instances):
        self.data_source = data_source
        self.batch_size = batch_size
        self.num_instances = num_instances
        self.num_pids_per_batch = self.batch_size // self.num_instances
        self.index_dic = defaultdict(list)
        for index, (_, pid, _) in enumerate(self.data_source):
            self.index_dic[pid].append(index)
        self.pids = list(self.index_dic.keys())

        # estimate number of examples in an epoch
        self.length = 0
        for pid in self.pids:
            idxs = self.index_dic[pid]
            num = len(idxs)
            if num < self.num_instances:
                num = self.num_instances
            self.length += num - num % self.num_instances

    def __iter__(self):
        batch_idxs_dict = defaultdict(list)

        for pid in self.pids:
            idxs = copy.deepcopy(self.index_dic[pid])
            if len(idxs) < self.num_instances:
                idxs = np.random.choice(idxs, size=self.num_instances, replace=True)
            random.shuffle(idxs)
            batch_idxs = []
            for idx in idxs:
                batch_idxs.append(idx)
                if len(batch_idxs) == self.num_instances:
                    batch_idxs_dict[pid].append(batch_idxs)
                    batch_idxs = []

        avai_pids = copy.deepcopy(self.pids)
        final_idxs = []

        while len(avai_pids) >= self.num_pids_per_batch:
            selected_pids = random.sample(avai_pids, self.num_pids_per_batch)
            for pid in selected_pids:
                batch_idxs = batch_idxs_dict[pid].pop(0)
                final_idxs.extend(batch_idxs)
                if len(batch_idxs_dict[pid]) == 0:
                    avai_pids.remove(pid)

        self.length = len(final_idxs)
        return iter(final_idxs)

    def __len__(self):
        return self.length


def get_data_provider(opt, dataset_name, dataset_root):
    num_gpus = (len(opt.network.gpus) + 1) // 2
    test_batch_size = opt.test.batch_size * num_gpus
   
    # data augmenter
    random_mirror = opt.aug.get('random_mirror', False)
    random_erase = opt.aug.get('random_erase', False)
    pad = opt.aug.get('pad', False)
    random_crop = opt.aug.get('random_crop', False)

    h, w = opt.aug.resize_size
    def get_train_aug(random_mirror=True, pad=10, random_crop=True, random_erase=True):
        train_aug = list()
        train_aug.append(T.Resize((h, w)))
        if random_mirror:
            train_aug.append(T.RandomHorizontalFlip())
        if pad:
            train_aug.append(T.Pad(padding=pad))
        if random_crop:
            train_aug.append(T.RandomCrop((h, w)))
    
        train_aug.append(T.ToTensor())
        if random_erase:
            train_aug.append(RandomErasing(probability = 1, mean=[0.485, 0.456, 0.406]))
        train_aug.append(T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
        train_aug = T.Compose(train_aug)
        return train_aug
    train_aug = get_train_aug(random_erase=True)

    test_aug = list()
    test_aug.append(T.Resize((h, w)))
    test_aug.append(T.ToTensor())
    test_aug.append(T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
    test_aug = T.Compose(test_aug)

    dataset = init_dataset(dataset_name, dataset_root)
    pids = [a[1] for a in dataset.train]
    num_class = np.max(pids) + 1
         
    train_set = ImageData(dataset.train, train_aug)
    test_set = ImageData(dataset.query + dataset.gallery, test_aug)

    if opt.train.sampler == 'softmax':
        train_batch_size = opt.train.batch_size * num_gpus
        train_loader = DataLoader(train_set, batch_size=train_batch_size, shuffle=True,
                                  num_workers=opt.network.workers, pin_memory=True, drop_last=True)
    elif opt.train.sampler == 'triplet':
        train_batch_size = opt.train.batch_size
        train_loader = DataLoader(train_set, batch_size=train_batch_size,
                                  sampler=RandomIdentitySampler(dataset.train, opt.train.batch_size, opt.train.k_size),
                                  num_workers=opt.network.workers, pin_memory=True)
    else:
        raise ValueError('sampler must be softmax or triplet, but get {}'.format(opt.train.sampler))

    test_loader = DataLoader(test_set, batch_size=test_batch_size, num_workers=opt.network.workers, pin_memory=True)
    return train_loader, test_loader, len(dataset.query), num_class  # return number of query


if __name__ == "__main__":
    from config import opt

    train_loader, test_loader, num_query = get_data_provider(opt)
    from IPython import embed

    embed()
