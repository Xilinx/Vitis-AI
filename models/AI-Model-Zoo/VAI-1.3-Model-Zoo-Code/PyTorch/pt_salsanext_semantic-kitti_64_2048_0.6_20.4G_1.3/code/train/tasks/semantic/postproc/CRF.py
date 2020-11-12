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

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class LocallyConnectedXYZLayer(nn.Module):
    def __init__(self, h, w, sigma, nclasses):
        super().__init__()
        # size of window
        self.h = h
        self.padh = h // 2
        self.w = w
        self.padw = w // 2
        assert (self.h % 2 == 1 and self.w % 2 == 1)  # window must be odd
        self.sigma = sigma
        self.gauss_den = 2 * self.sigma ** 2
        self.nclasses = nclasses

    def forward(self, xyz, softmax, mask):
        # softmax size
        N, C, H, W = softmax.shape

        # make sofmax zero everywhere input is invalid
        softmax = softmax * mask.unsqueeze(1).float()

        # get x,y,z for distance (shape N,1,H,W)
        x = xyz[:, 0].unsqueeze(1)
        y = xyz[:, 1].unsqueeze(1)
        z = xyz[:, 2].unsqueeze(1)

        # im2col in size of window of input (x,y,z separately)
        window_x = F.unfold(x, kernel_size=(self.h, self.w),
                            padding=(self.padh, self.padw))
        center_x = F.unfold(x, kernel_size=(1, 1),
                            padding=(0, 0))
        window_y = F.unfold(y, kernel_size=(self.h, self.w),
                            padding=(self.padh, self.padw))
        center_y = F.unfold(y, kernel_size=(1, 1),
                            padding=(0, 0))
        window_z = F.unfold(z, kernel_size=(self.h, self.w),
                            padding=(self.padh, self.padw))
        center_z = F.unfold(z, kernel_size=(1, 1),
                            padding=(0, 0))

        # sq distance to center (center distance is zero)
        unravel_dist2 = (window_x - center_x) ** 2 + \
                        (window_y - center_y) ** 2 + \
                        (window_z - center_z) ** 2

        # weight input distance by gaussian weights
        unravel_gaussian = torch.exp(- unravel_dist2 / self.gauss_den)

        # im2col in size of window of softmax to reweight by gaussian weights from input
        cloned_softmax = softmax.clone()
        for i in range(self.nclasses):
            # get the softmax for this class
            c_softmax = softmax[:, i].unsqueeze(1)
            # unfold this class to weigh it by the proper gaussian weights
            unravel_softmax = F.unfold(c_softmax,
                                       kernel_size=(self.h, self.w),
                                       padding=(self.padh, self.padw))
            unravel_w_softmax = unravel_softmax * unravel_gaussian
            # add dimenssion 1 to obtain the new softmax for this class
            unravel_added_softmax = unravel_w_softmax.sum(dim=1).unsqueeze(1)
            # fold it and put it in new tensor
            added_softmax = unravel_added_softmax.view(N, H, W)
            cloned_softmax[:, i] = added_softmax

        return cloned_softmax


class CRF(nn.Module):
    def __init__(self, params, nclasses):
        super().__init__()
        self.params = params
        self.iter = torch.nn.Parameter(torch.tensor(params["iter"]),
                                       requires_grad=False)
        self.lcn_size = torch.nn.Parameter(torch.tensor([params["lcn_size"]["h"],
                                                         params["lcn_size"]["w"]]),
                                           requires_grad=False)
        self.xyz_coef = torch.nn.Parameter(torch.tensor(params["xyz_coef"]),
                                           requires_grad=False).float()
        self.xyz_sigma = torch.nn.Parameter(torch.tensor(params["xyz_sigma"]),
                                            requires_grad=False).float()

        self.nclasses = nclasses
        print("Using CRF!")

        # define layers here
        # compat init
        self.compat_kernel_init = np.reshape(np.ones((self.nclasses, self.nclasses)) -
                                             np.identity(self.nclasses),
                                             [self.nclasses, self.nclasses, 1, 1])

        # bilateral compatibility matrixes
        self.compat_conv = nn.Conv2d(self.nclasses, self.nclasses, 1)
        self.compat_conv.weight = torch.nn.Parameter(torch.from_numpy(
            self.compat_kernel_init).float() * self.xyz_coef, requires_grad=True)

        # locally connected layer for message passing
        self.local_conn_xyz = LocallyConnectedXYZLayer(params["lcn_size"]["h"],
                                                       params["lcn_size"]["w"],
                                                       params["xyz_coef"],
                                                       self.nclasses)

    def forward(self, input, softmax, mask):
        # use xyz
        xyz = input[:, 1:4]

        # iteratively
        for iter in range(self.iter):
            # message passing as locally connected layer
            locally_connected = self.local_conn_xyz(xyz, softmax, mask)

            # reweigh with the 1x1 convolution
            reweight_softmax = self.compat_conv(locally_connected)

            # add the new values to the original softmax
            reweight_softmax = reweight_softmax + softmax

            # lastly, renormalize
            softmax = F.softmax(reweight_softmax, dim=1)

        return softmax
