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

#!/usr/bin/env python3
# This file is covered by the LICENSE file in the root of this project.

import argparse
import os
import sys

import numpy as np
import torch
import yaml
from common.laserscan import SemLaserScan
from tasks.semantic.modules.ioueval import biouEval

# possible splits
splits = ["train", "valid", "test"]

if __name__ == '__main__':
    parser = argparse.ArgumentParser("./evaluate_biou.py")
    parser.add_argument(
        '--dataset', '-d',
        type=str,
        required=True,
        help='Dataset dir. No Default',
    )
    parser.add_argument(
        '--predictions', '-p',
        type=str,
        required=None,
        help='Prediction dir. Same organization as dataset, but predictions in'
             'each sequences "prediction" directory. No Default. If no option is set'
             ' we look for the labels in the same directory as dataset'
    )
    parser.add_argument(
        '--split', '-s',
        type=str,
        required=False,
        choices=["train", "valid", "test"],
        default="valid",
        help='Split to evaluate on. One of ' +
             str(splits) + '. Defaults to %(default)s',
    )
    parser.add_argument(
        '--data_cfg', '-dc',
        type=str,
        required=False,
        default="config/labels/semantic-kitti.yaml",
        help='Dataset config file. Defaults to %(default)s',
    )
    parser.add_argument(
        '--border', '-bs',
        type=int,
        required=False,
        default=1,
        help='Border size. Defaults to %(default)s',
    )
    parser.add_argument(
        '--conn', '-c',
        type=int,
        required=False,
        default=4,
        help='Kernel connectivity. Defaults to %(default)s',
    )
    FLAGS, unparsed = parser.parse_known_args()

    # fill in real predictions dir
    if FLAGS.predictions is None:
        FLAGS.predictions = FLAGS.dataset

    # print summary of what we will do
    print("*" * 80)
    print("INTERFACE:")
    print("Data: ", FLAGS.dataset)
    print("Predictions: ", FLAGS.predictions)
    print("Split: ", FLAGS.split)
    print("Config: ", FLAGS.data_cfg)
    print("Border Mask Size", FLAGS.border)
    print("Border Mask Connectivity", FLAGS.conn)
    print("*" * 80)

    # assert split
    assert (FLAGS.split in splits)

    # open data config file
    try:
        print("Opening data config file %s" % FLAGS.data_cfg)
        DATA = yaml.safe_load(open(FLAGS.data_cfg, 'r'))
    except Exception as e:
        print(e)
        print("Error opening data yaml file.")
        quit()

    # get number of interest classes, and the label mappings
    class_strings = DATA["labels"]
    class_remap = DATA["learning_map"]
    class_inv_remap = DATA["learning_map_inv"]
    class_ignore = DATA["learning_ignore"]
    nr_classes = len(class_inv_remap)

    # make lookup table for mapping
    maxkey = 0
    for key, data in class_remap.items():
        if key > maxkey:
            maxkey = key
    # +100 hack making lut bigger just in case there are unknown labels
    remap_lut = np.zeros((maxkey + 100), dtype=np.int32)
    for key, data in class_remap.items():
        try:
            remap_lut[key] = data
        except IndexError:
            print("Wrong key ", key)
    # print(remap_lut)

    # create evaluator
    ignore = []
    for cl, ign in class_ignore.items():
        if ign:
            x_cl = int(cl)
            ignore.append(x_cl)
            print("Ignoring xentropy class ", x_cl, " in IoU evaluation")

    # create evaluator
    device = torch.device(
        "cuda") if torch.cuda.is_available() else torch.device('cpu')
    evaluator = biouEval(nr_classes, device, ignore,
                         FLAGS.border, FLAGS.conn)
    evaluator.reset()

    # get test set
    test_sequences = DATA["split"][FLAGS.split]

    # get scan paths
    scan_names = []
    for sequence in test_sequences:
        sequence = '{0:02d}'.format(int(sequence))
        scan_paths = os.path.join(FLAGS.dataset, "sequences",
                                  str(sequence), "velodyne")
        # populate the scan names
        seq_scan_names = [os.path.join(dp, f) for dp, dn, fn in os.walk(
            os.path.expanduser(scan_paths)) for f in fn if ".bin" in f]
        seq_scan_names.sort()
        scan_names.extend(seq_scan_names)
    # print(scan_names)

    # get label paths
    label_names = []
    for sequence in test_sequences:
        sequence = '{0:02d}'.format(int(sequence))
        label_paths = os.path.join(FLAGS.dataset, "sequences",
                                   str(sequence), "labels")
        # populate the label names
        seq_label_names = [os.path.join(dp, f) for dp, dn, fn in os.walk(
            os.path.expanduser(label_paths)) for f in fn if ".label" in f]
        seq_label_names.sort()
        label_names.extend(seq_label_names)
    # print(label_names)

    # get predictions paths
    pred_names = []
    for sequence in test_sequences:
        sequence = '{0:02d}'.format(int(sequence))
        pred_paths = os.path.join(FLAGS.predictions, "sequences",
                                  sequence, "predictions")
        # populate the label names
        seq_pred_names = [os.path.join(dp, f) for dp, dn, fn in os.walk(
            os.path.expanduser(pred_paths)) for f in fn if ".label" in f]
        seq_pred_names.sort()
        pred_names.extend(seq_pred_names)
    # print(pred_names)

    # check that I have the same number of files
    # print("labels: ", len(label_names))
    # print("predictions: ", len(pred_names))
    assert (len(label_names) == len(scan_names) and
            len(label_names) == len(pred_names))

    print("Evaluating sequences: ")
    # open each file, get the tensor, and make the iou comparison
    for scan_file, label_file, pred_file in zip(scan_names, label_names, pred_names):
        print("evaluating label ", label_file, "with", pred_file)
        # open label
        label = SemLaserScan(project=True)
        label.open_scan(scan_file)
        label.open_label(label_file)
        u_label_sem = remap_lut[label.sem_label]  # remap to xentropy format
        p_label_sem = remap_lut[label.proj_sem_label]  # remap to xentropy format
        u_scan_px = label.proj_x
        u_scan_py = label.proj_y

        # open prediction
        pred = SemLaserScan(project=True)
        pred.open_scan(scan_file)
        pred.open_label(pred_file)
        u_pred_sem = remap_lut[pred.sem_label]  # remap to xentropy format

        # add single scan to evaluation
        evaluator.addBorderBatch1d(p_label_sem, u_pred_sem, u_label_sem,
                                   u_scan_px, u_scan_py)

    # when I am done, print the evaluation
    m_accuracy = evaluator.getacc()
    m_jaccard, class_jaccard = evaluator.getIoU()

    print('Validation set:\n'
          'bAcc avg {m_accuracy:.3f}\n'
          'bIoU avg {m_jaccard:.3f}'.format(m_accuracy=m_accuracy,
                                            m_jaccard=m_jaccard))
    # print also classwise
    for i, jacc in enumerate(class_jaccard):
        if i not in ignore:
            print('bIoU class {i:} [{class_str:}] = {jacc:.3f}'.format(
                i=i, class_str=class_strings[class_inv_remap[i]], jacc=jacc))

    # print for spreadsheet
    print("*" * 80)
    print("below can be copied straight for paper table")
    for i, jacc in enumerate(class_jaccard):
        if i not in ignore:
            sys.stdout.write('{jacc:.3f}'.format(jacc=jacc.item()))
            sys.stdout.write(",")
    sys.stdout.write('{jacc:.3f}'.format(jacc=m_jaccard.item()))
    sys.stdout.write(",")
    sys.stdout.write('{acc:.3f}'.format(acc=m_accuracy.item()))
    sys.stdout.write('\n')
    sys.stdout.flush()
