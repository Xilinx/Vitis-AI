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

import glob
import re
import os
from os import path as osp
from ipdb import set_trace

"""Dataset classes"""


class FaceReid(object):
    """
    Dataset statistics:
    # identities: 2761+2761(+1 for background)
    # images: 34323(train) + 5412(query) + 28228(gallery)
    """
    def __init__(self, dataset_dir='./data/face_reid', pid_start=0):
        self.dataset_dir = dataset_dir
        self.train_dir = osp.join(self.dataset_dir, 'train')
        self.query_dir = osp.join(self.dataset_dir, 'test')
        self.gallery_dir = osp.join(self.dataset_dir, 'test')
        self.pid_start = pid_start

        self._check_before_run()

        train, num_train_pids, num_train_imgs = self._process_dir(self.train_dir, 'train')
        query, num_query_pids, num_query_imgs = self._process_dir(self.query_dir, 'query')
        gallery, num_gallery_pids, num_gallery_imgs = self._process_dir(self.gallery_dir, 'gallery')
        num_total_pids = num_train_pids + num_query_pids
        num_total_imgs = num_train_imgs + num_query_imgs + num_gallery_imgs

        print("=> Face_reid loaded")
        print("Dataset statistics:")
        print("  ------------------------------")
        print("  subset   | # ids | # images")
        print("  ------------------------------")
        print("  train    | {:5d} | {:8d}".format(num_train_pids, num_train_imgs))
        print("  query    | {:5d} | {:8d}".format(num_query_pids, num_query_imgs))
        print("  gallery  | {:5d} | {:8d}".format(num_gallery_pids, num_gallery_imgs))
        print("  ------------------------------")
        print("  total    | {:5d} | {:8d}".format(num_total_pids, num_total_imgs))
        print("  ------------------------------")

        self.train = train
        self.query = query
        self.gallery = gallery

        self.num_train_pids = num_train_pids
        self.num_query_pids = num_query_pids
        self.num_gallery_pids = num_gallery_pids

    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not osp.exists(self.dataset_dir):
            raise RuntimeError("'{}' is not available".format(self.dataset_dir))
        if not osp.exists(self.train_dir):
            raise RuntimeError("'{}' is not available".format(self.train_dir))
        if not osp.exists(self.query_dir):
            raise RuntimeError("'{}' is not available".format(self.query_dir))
        if not osp.exists(self.gallery_dir):
            raise RuntimeError("'{}' is not available".format(self.gallery_dir))

    def _process_dir(self, dir_path, subset):
        img_list_file = os.path.join(self.dataset_dir, '%s_list.txt'%subset)

        dataset = []
        faceid_container = set()
        with open(img_list_file) as f:
            lines = f.readlines()
        for line in lines:
            img_path, face_id = line.split(',')
            img_path = os.path.join(dir_path, img_path)
            face_id = int(face_id[:-1])
            dataset.append((img_path, face_id, 0)) # dataset format: img_path, face_id, camera_id=0
            faceid_container.add(face_id)

        num_ids = len(faceid_container)
        num_imgs = len(lines)
        return dataset, num_ids, num_imgs

if __name__ == '__main__':
    dataset = FaceReid()
    set_trace()
