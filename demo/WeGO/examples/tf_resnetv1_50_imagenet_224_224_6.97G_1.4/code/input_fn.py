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
import sys

import numpy as np
import tensorflow as tf
from tqdm import tqdm
from dataloader import DataLoader


def calib_input(preprocess_type, input_height, input_width, eval_image_list, eval_steps, eval_batch, eval_iter, eval_image_path, label_offset):
    batch_group = []
    batch_group_labels = []
    with open(eval_image_list, 'r') as fr:
        lines = fr.readlines()
    with tf.compat.v1.Session() as sess:
        data_loader = DataLoader(input_height, input_width)
        image, input_plhd = data_loader.build_preprocess(
            style=preprocess_type)
        for i in tqdm(range(eval_steps)):
            batch_images = []
            batch_labels = []
            start_idx = i * eval_batch
            end_idx = (i + 1) * eval_batch
            eval_batch_ = eval_batch if (end_idx <= eval_iter) else (
                eval_iter - start_idx)
            for j in range(eval_batch_):
                idx = start_idx + j
                line = lines[idx]
                img_path, label = line.strip().split(" ")
                img_path = os.path.join(eval_image_path, img_path)
                label = int(label) + 1 - label_offset
                label = np.array([label], dtype=np.int64)
                if not os.path.exists(img_path + ".npy"):
                    image_val = sess.run(
                        image, feed_dict={input_plhd: img_path})
                    np.save(img_path, image_val)
                image_val = np.load(img_path+".npy")
                batch_images.append(image_val)
                batch_labels.append(label)
            batch_images = np.squeeze(batch_images)
            batch_group.append(batch_images)
            batch_group_labels.append(batch_labels)
    return batch_group, batch_group_labels
