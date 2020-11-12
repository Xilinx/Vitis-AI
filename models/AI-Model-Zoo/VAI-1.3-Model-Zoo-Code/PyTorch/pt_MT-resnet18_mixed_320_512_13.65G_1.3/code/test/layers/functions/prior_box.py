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

# PART OF THIS FILE AT ALL TIMES.

from __future__ import division
from math import sqrt as sqrt
from itertools import product as product
import torch


class PriorBox(object):
    """Compute priorbox coordinates in center-offset form for each source
    feature map.
    """
    def __init__(self, cfg):
        super(PriorBox, self).__init__()
        self.image_size = cfg['resize']
        # number of priors for feature map location (either 4 or 6)
        self.num_priors = len(cfg['aspect_ratios'])
        self.variance = cfg['variance'] or [0.1]
        self.feature_maps = cfg['feature_maps']
        self.min_sizes = cfg['min_sizes']
        self.max_sizes = cfg['max_sizes']
        self.steps = cfg['steps']
        self.aspect_ratios = cfg['aspect_ratios']
        self.clip = cfg['clip']
        for v in self.variance:
            if v <= 0:
                raise ValueError('Variances must be greater than 0')

    def forward(self):
        mean = []
        for k, f in enumerate(self.feature_maps):
            for i, j in product(range(f[0]), range(f[1])):
                f_kx = self.image_size[1] / self.steps[k]
                f_ky = self.image_size[0] / self.steps[k]
                # unit center x,y
                cx = (j + 0.5) / f_kx
                cy = (i + 0.5) / f_ky

                # aspect_ratio: 1
                # rel size: min_size
                s_kx = self.min_sizes[k]/self.image_size[1]
                s_ky = self.min_sizes[k]/self.image_size[0]
                mean += [cx, cy, s_kx, s_ky]

                # aspect_ratio: 1
                # rel size: sqrt(s_k * s_(k+1))
                s_k_primex = sqrt(s_kx * (self.max_sizes[k]/self.image_size[1]))
                s_k_primey = sqrt(s_ky * (self.max_sizes[k]/self.image_size[0]))
                mean += [cx, cy, s_k_primex, s_k_primey]

                # rest of aspect ratios
                for ar in self.aspect_ratios[k]:
                    mean += [cx, cy, s_kx*sqrt(ar), s_ky/sqrt(ar)]
                    mean += [cx, cy, s_kx/sqrt(ar), s_ky*sqrt(ar)]
        # back to torch land
        output = torch.Tensor(mean).view(-1, 4)
        if self.clip:
            output.clamp_(max=1, min=0)
        return output
