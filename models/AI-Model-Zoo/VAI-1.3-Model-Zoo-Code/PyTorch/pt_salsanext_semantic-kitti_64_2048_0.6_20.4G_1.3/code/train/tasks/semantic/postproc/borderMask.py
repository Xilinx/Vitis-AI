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
"""Border Mask for 2D labeled range images.

Simple module to obtain the border mask of a given range image.

The border mask is defined as the zone where are intersections between
differrent classes for the given range image.

In this case we will violate a little bit the definition and will augment it. We
define the border mask as the zone where are intersections between differnet
classes for the given range image in determined neighborhood. To obtain this
border mask we will need to apply de binary erosion algorithm multiple times to
the same range image.

Example:
  Suppose we have 3 classes and this given range image(labeled):
  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
  [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
  [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
  [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
  [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
  [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
  [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
  [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
  [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
  [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
  [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 2, 2, 2, 0, 0],
  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 2, 2, 2, 0, 0],
  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 2, 2, 2, 0, 0],
  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 2, 2, 2, 0, 0],
  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 2, 2, 2, 0, 0],
  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

  The output of the bordermask would like:
  # 1 erode iteration with a connectivity kernel of 4:
  [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
  [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
  [1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1],
  [1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1],
  [1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1],
  [1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1],
  [1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1],
  [1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1],
  [1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1],
  [1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1],
  [1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1],
  [1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1],
  [1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1],
  [1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
  [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1],
  [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1],
  [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1],
  [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
  [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 1],
  [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]

  # 2 erode iterations with a connectivity kernel of 8:
  [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
  [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
  [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1],
  [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1],
  [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1],
  [1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 1, 1],
  [1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 1, 1],
  [1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 1, 1],
  [1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 1, 1],
  [1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 1, 1],
  [1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 1, 1],
  [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
  [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
  [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
  [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
  [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1],
  [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1],
  [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1],
  [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
  [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from common.onehot import oneHot


class borderMask(nn.Module):
    def __init__(self, nclasses, device, border_size, kern_conn=4, background_class=None):
        """Get the binary border mask of a labeled 2d range image.

      Args:
          nclasses(int)         : The number of classes labeled in the input image
          device(torch.device)  : Process in host or cuda?
          border_size(int)      : How many erode iterations to perform for the mask
          kern_conn(int)        : The connectivity kernel number (4 or 8)
          background_class(int) : "unlabeled" class in dataset (to avoid double borders)

      Returns:
          eroded_output(tensor) : The 2d binary border mask, 1 where a intersection
                                  between classes occurs, 0 everywhere else

      """
        super().__init__()
        self.nclasses = nclasses
        self.device = device
        self.border_size = border_size
        self.kern_conn = kern_conn
        self.background_class = background_class
        if self.background_class is not None:
            self.include_idx = list(range(self.nclasses))
            self.exclude_idx = self.include_idx.pop(self.background_class)

        # check connectivity
        # For obtaining the border mask we will be eroding the input image, for this
        # reason we only support erode_kernels with connectivity 4 or 8
        assert self.kern_conn in (4, 8), ("The specified kernel connectivity(kern_conn= %r) is "
                                          "not supported" % self.kern_conn)

        # make the onehot inferer
        self.onehot = oneHot(self.device,
                             self.nclasses,
                             spatial_dim=2)  # range labels

    def forward(self, range_label):
        # length of shape of range_label must be 3 (N, H, W)
        must_unbatch = False  # remove batch dimension after operation?
        if len(range_label.shape) != 3:
            range_label = range_label[None, ...]
            must_unbatch = True

        # The range_label comes labeled, we need to create one tensor per class, thus:
        input_tensor = self.onehot(range_label)  # (N, C, H, W)

        # Because we are using GT range_labels, there is a lot of pixels that end up
        # unlabeled(thus, in the background). If we feed the erosion algorithm with
        # this "raw" gt_labels we will detect intersection between the other classes
        # and the backgorund, and we will end with the incorrect border mask. To solve
        # this issue we need to pre process the input gt_label. The artifact in this
        # case will be to sum the background channel(mostly the channel 0) to
        # all the rest channels expect for the background channel itself.
        # This will allow us to avoid detecting intersections between a class and the
        # background. This also force us to change the logical AND we were doing to
        # obtain the border mask when we were working with predicted labels.
        # With predicted labels won't see this problem because all the pixels belongs
        # to at least one class
        if self.background_class is not None:
            input_tensor[:, self.include_idx] = input_tensor[:, self.include_idx] + \
                                                input_tensor[:, self.exclude_idx]

        # C denotes a number of channels, N, H and W are dismissed
        C = input_tensor.shape[1]

        # Create an empty erode kernel and send it to 'device'
        erode_kernel = torch.zeros((C, 1, 3, 3), device=self.device)
        if self.kern_conn == 4:
            erode_kernel[:] = torch.tensor([[0, 1, 0],
                                            [1, 1, 1],
                                            [0, 1, 0]], device=self.device)
        else:
            erode_kernel[:] = torch.tensor([[1, 1, 1],
                                            [1, 1, 1],
                                            [1, 1, 1]], device=self.device)

        # to check connectivity
        kernel_sum = erode_kernel[0][0].sum()  # should be kern_conn + 1

        # erode the input image border_size times
        erode_input = input_tensor
        for _ in range(self.border_size):
            eroded_output = F.conv2d(erode_input, erode_kernel, groups=C, padding=1)
            # Pick the elements that match the kernel_sum to obtain the eroded
            # output and convert to dtype=float32
            eroded_output = (eroded_output == kernel_sum).float()
            erode_input = eroded_output

        # We want to sum up all the channels into 1 unique border mask
        # Even when we added the background to all the rest of the channels, there
        # might be "bodies" in the background channel, thus, the erosion process can
        # output "false positives" were this "bodies" are present in the background.
        # We need to obtain the background mask and add it to the eroded bodies to
        # obtain a consisent output once we calculate the border mask
        if self.background_class is not None:
            background_mask = (eroded_output[:, self.exclude_idx] == 1)

        # The eroded_bodies mask will consist in all the pixels were the convolution
        # returned 1 for all the channels, therefore we need to sum up all the
        # channels into one unique tensor and add the background mask to avoid having
        # the background in the border mask output
        eroded_bodies = (eroded_output.sum(1, keepdim=True) == 1)
        if self.background_class is not None:
            eroded_bodies = eroded_bodies + background_mask

        # we want the opposite
        borders = 1 - eroded_bodies

        # unbatch?
        if must_unbatch:
            borders = borders[0]
            # import cv2
            # import numpy as np
            # bordersprint = (borders * 255).squeeze().cpu().numpy().astype(np.uint8)
            # cv2.imshow("border", bordersprint)
            # cv2.waitKey(0)

        return borders


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser("./borderMask.py")
    parser.add_argument(
        '--scan', '-s',
        type=str,
        required=True,
        help='Scan to get xyz. No Default',
    )
    parser.add_argument(
        '--label', '-l',
        type=str,
        required=True,
        help='Label to calculate border mask. No Default',
    )
    parser.add_argument(
        '--exclude_class', '-e',
        type=int,
        required=False,
        default=None,
        help='Label to ignore. No Default',
    )
    parser.add_argument(
        '--border', '-b',
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

    # print summary of what we will do
    print("----------")
    print("INTERFACE:")
    print("Scan", FLAGS.scan)
    print("Label", FLAGS.label)
    print("Exclude class", FLAGS.exclude_class)
    print("Border", FLAGS.border)
    print("Connectivity", FLAGS.conn)
    print("----------\n")

    # get device
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    # define the border mask
    bm = borderMask(300, device, FLAGS.border,
                    FLAGS.conn, FLAGS.exclude_class)

    # imports for inference part
    import cv2
    import numpy as np
    from common.laserscan import SemLaserScan

    # open label and project
    scan = SemLaserScan(project=True, max_classes=300)
    scan.open_scan(FLAGS.scan)
    scan.open_label(FLAGS.label)

    # get the things I need
    proj_range = torch.from_numpy(scan.proj_range).to(device)
    proj_sem_label = torch.from_numpy(scan.proj_sem_label).long().to(device)
    proj_sem_color = torch.from_numpy(scan.proj_sem_color).to(device)

    # run the border mask
    border_mask = bm(proj_sem_label)

    # bring to numpy and normalize for showing
    proj_range = proj_range.cpu().numpy()
    proj_sem_label = proj_sem_label.cpu().numpy()
    proj_sem_color = proj_sem_color.cpu().numpy()
    border_mask = border_mask.cpu().numpy().squeeze()

    # norm
    proj_range = (proj_range / proj_range.max() * 255).astype(np.uint8)
    border_mask = (border_mask * 255).astype(np.uint8)

    # show
    cv2.imshow("range", proj_range)
    cv2.imshow("label", proj_sem_color)
    cv2.imshow("border", border_mask)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
