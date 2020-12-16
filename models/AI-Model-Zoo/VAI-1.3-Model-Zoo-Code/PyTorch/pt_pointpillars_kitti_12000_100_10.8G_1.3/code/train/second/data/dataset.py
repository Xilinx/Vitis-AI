
# This code is based on: https://github.com/nutonomy/second.pytorch.git
# 
# MIT License
# Copyright (c) 2018 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import pathlib
import pickle
import time
from functools import partial

import numpy as np

from second.core import box_np_ops
from second.core import preprocess as prep
from second.data import kitti_common as kitti
from second.data.preprocess import _read_and_prep_v9


class Dataset(object):
    """An abstract class representing a pytorch-like Dataset.
    All other datasets should subclass it. All subclasses should override
    ``__len__``, that provides the size of the dataset, and ``__getitem__``,
    supporting integer indexing in range from 0 to len(self) exclusive.
    """

    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError



class KittiDataset(Dataset):
    def __init__(self, info_path, root_path, num_point_features,
                 target_assigner, feature_map_size, prep_func):
        with open(info_path, 'rb') as f:
            infos = pickle.load(f)
        #self._kitti_infos = kitti.filter_infos_by_used_classes(infos, class_names)
        self._root_path = root_path
        self._kitti_infos = infos
        self._num_point_features = num_point_features
        print("remain number of infos:", len(self._kitti_infos))
        # generate anchors cache
        # [352, 400]
        ret = target_assigner.generate_anchors(feature_map_size)
        anchors = ret["anchors"]
        anchors = anchors.reshape([-1, 7])
        matched_thresholds = ret["matched_thresholds"]
        unmatched_thresholds = ret["unmatched_thresholds"]
        anchors_bv = box_np_ops.rbbox2d_to_near_bbox(
            anchors[:, [0, 1, 3, 4, 6]])
        anchor_cache = {
            "anchors": anchors,
            "anchors_bv": anchors_bv,
            "matched_thresholds": matched_thresholds,
            "unmatched_thresholds": unmatched_thresholds,
        }
        self._prep_func = partial(prep_func, anchor_cache=anchor_cache)

    def __len__(self):
        return len(self._kitti_infos)

    @property
    def kitti_infos(self):
        return self._kitti_infos

    def __getitem__(self, idx):
        return _read_and_prep_v9(
            info=self._kitti_infos[idx],
            root_path=self._root_path,
            num_point_features=self._num_point_features,
            prep_func=self._prep_func)
