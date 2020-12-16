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

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def get_gaussian_kernel(kernel_size=3, sigma=2, channels=1):
    # Create a x, y coordinate grid of shape (kernel_size, kernel_size, 2)
    x_coord = torch.arange(kernel_size)
    x_grid = x_coord.repeat(kernel_size).view(kernel_size, kernel_size)
    y_grid = x_grid.t()
    xy_grid = torch.stack([x_grid, y_grid], dim=-1).float()

    mean = (kernel_size - 1) / 2.
    variance = sigma ** 2.

    # Calculate the 2-dimensional gaussian kernel which is
    # the product of two gaussian distributions for two different
    # variables (in this case called x and y)
    gaussian_kernel = (1. / (2. * math.pi * variance)) * \
                      torch.exp(-torch.sum((xy_grid - mean) ** 2., dim=-1) / (2 * variance))

    # Make sure sum of values in gaussian kernel equals 1.
    gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)

    # Reshape to 2d depthwise convolutional weight
    gaussian_kernel = gaussian_kernel.view(kernel_size, kernel_size)

    return gaussian_kernel


class KNN(nn.Module):
    def __init__(self, params, nclasses):
        super().__init__()
        print("*" * 80)
        print("Cleaning point-clouds with kNN post-processing")
        self.knn = params["knn"]
        self.search = params["search"]
        self.sigma = params["sigma"]
        self.cutoff = params["cutoff"]
        self.nclasses = nclasses
        print("kNN parameters:")
        print("knn:", self.knn)
        print("search:", self.search)
        print("sigma:", self.sigma)
        print("cutoff:", self.cutoff)
        print("nclasses:", self.nclasses)
        print("*" * 80)

    def forward(self, proj_range, unproj_range, proj_argmax, px, py):
        ''' Warning! Only works for un-batched pointclouds.
            If they come batched we need to iterate over the batch dimension or do
            something REALLY smart to handle unaligned number of points in memory
        '''
        # get device
        if proj_range.is_cuda:
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")

        # sizes of projection scan
        H, W = proj_range.shape

        # number of points
        P = unproj_range.shape

        # check if size of kernel is odd and complain
        if (self.search % 2 == 0):
            raise ValueError("Nearest neighbor kernel must be odd number")

        # calculate padding
        pad = int((self.search - 1) / 2)

        # unfold neighborhood to get nearest neighbors for each pixel (range image)
        proj_unfold_k_rang = F.unfold(proj_range[None, None, ...],
                                      kernel_size=(self.search, self.search),
                                      padding=(pad, pad))

        # index with px, py to get ALL the pcld points
        idx_list = py * W + px
        unproj_unfold_k_rang = proj_unfold_k_rang[:, :, idx_list]

        # WARNING, THIS IS A HACK
        # Make non valid (<0) range points extremely big so that there is no screwing
        # up the nn self.search
        unproj_unfold_k_rang[unproj_unfold_k_rang < 0] = float("inf")

        # now the matrix is unfolded TOTALLY, replace the middle points with the actual range points
        center = int(((self.search * self.search) - 1) / 2)
        unproj_unfold_k_rang[:, center, :] = unproj_range

        # now compare range
        k2_distances = torch.abs(unproj_unfold_k_rang - unproj_range)

        # make a kernel to weigh the ranges according to distance in (x,y)
        # I make this 1 - kernel because I want distances that are close in (x,y)
        # to matter more
        inv_gauss_k = (
                1 - get_gaussian_kernel(self.search, self.sigma, 1)).view(1, -1, 1)
        inv_gauss_k = inv_gauss_k.to(device).type(proj_range.type())

        # apply weighing
        k2_distances = k2_distances * inv_gauss_k

        # find nearest neighbors
        _, knn_idx = k2_distances.topk(
            self.knn, dim=1, largest=False, sorted=False)

        # do the same unfolding with the argmax
        proj_unfold_1_argmax = F.unfold(proj_argmax[None, None, ...].float(),
                                        kernel_size=(self.search, self.search),
                                        padding=(pad, pad)).long()
        unproj_unfold_1_argmax = proj_unfold_1_argmax[:, :, idx_list]

        # get the top k predictions from the knn at each pixel
        knn_argmax = torch.gather(
            input=unproj_unfold_1_argmax, dim=1, index=knn_idx)

        # fake an invalid argmax of classes + 1 for all cutoff items
        if self.cutoff > 0:
            knn_distances = torch.gather(input=k2_distances, dim=1, index=knn_idx)
            knn_invalid_idx = knn_distances > self.cutoff
            knn_argmax[knn_invalid_idx] = self.nclasses

        # now vote
        # argmax onehot has an extra class for objects after cutoff
        knn_argmax_onehot = torch.zeros(
            (1, self.nclasses + 1, P[0]), device=device).type(proj_range.type())
        ones = torch.ones_like(knn_argmax).type(proj_range.type())
        knn_argmax_onehot = knn_argmax_onehot.scatter_add_(1, knn_argmax, ones)

        # now vote (as a sum over the onehot shit)  (don't let it choose unlabeled OR invalid)
        knn_argmax_out = knn_argmax_onehot[:, 1:-1].argmax(dim=1) + 1

        # reshape again
        knn_argmax_out = knn_argmax_out.view(P)

        return knn_argmax_out
