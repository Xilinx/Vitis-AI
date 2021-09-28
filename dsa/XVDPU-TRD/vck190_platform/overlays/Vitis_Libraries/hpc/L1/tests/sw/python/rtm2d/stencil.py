#
# Copyright 2019 Xilinx, Inc.
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
#
import numpy as np
import argparse
import os
import pdb


class Stencil:
    def __init__(
            self,
            shape,
            order,
            dz=7.62,
            dx=7.62,
            dt=0.001,
            dtype=np.float32):
        self.shape = shape
        self.dtype = dtype
        self.m_z = shape[0]
        self.m_x = shape[1]

        self.dx = dx
        self.dz = dz
        self.dt = dt

        self.order = order
        self.halfOrder = self.order // 2
        self.computeCoef()

    def randImg(self):
        return np.random.rand(self.m_z, self.m_x).astype(self.dtype)

    def setV(self, v2):
        self.v2dt2 = v2 * self.dt * self.dt

    def computeCoef(self, dx=7.62, dz=7.62):
        self.coefx = np.zeros(self.order + 1, dtype=self.dtype)
        self.coefz = np.zeros(self.order + 1, dtype=self.dtype)
        if(self.order == 2):
            self.coefx[0] = self.coefz[0] = 1
            self.coefx[1] = self.coefz[1] = -2
            self.coefx[2] = self.coefz[2] = 1
        elif(self.order == 8):
            self.coefx[0] = self.coefz[0] = -1 / 560
            self.coefx[1] = self.coefz[1] = 8 / 315
            self.coefx[2] = self.coefz[2] = -1 / 5
            self.coefx[3] = self.coefz[3] = 8 / 5
            self.coefx[4] = self.coefz[4] = -205 / 72
            self.coefx[5] = self.coefz[5] = 8 / 5
            self.coefx[6] = self.coefz[6] = -1 / 5
            self.coefx[7] = self.coefz[7] = 8 / 315
            self.coefx[8] = self.coefz[8] = -1 / 560
        else:
            print("Order is not supported.")

        for i in range(self.order + 1):
            self.coefx[i] /= self.dx * self.dx
            self.coefz[i] /= self.dz * self.dz

    def setCoef(self, coefx, coefz):
        assert coefx.shape == coefz.shape
        self.order = coefx.size - 1
        self.halfOrder = self.order // 2
        self.coefx = coefx
        self.coefz = coefz

    def zero(self):
        return np.zeros(self.shape, dtype=self.dtype)

    def freelap(self, image):
        z, x = image.shape
        img = np.zeros([z, x], self.dtype)
        for k in range(1 + self.order):
            img[self.halfOrder: z - self.halfOrder,
                self.halfOrder: x - self.halfOrder] += self.coefx[k] * image[self.halfOrder: z - self.halfOrder,
                                                                             k: x - self.order + k] + self.coefz[k] * image[k: z - self.order + k,
                                                                                                                            self.halfOrder: x - self.halfOrder]
        return img

    def laplacian(self, image):
        assert(image.shape == self.shape)
        img = self.zero()
        for k in range(1 + self.order):
            img[self.halfOrder: self.m_z - self.halfOrder,
                self.halfOrder: self.m_x - self.halfOrder] += self.coefx[k] * image[self.halfOrder: self.m_z - self.halfOrder,
                                                                                    k: self.m_x - self.order + k] + self.coefz[k] * image[k: self.m_z - self.order + k,
                                                                                                                                          self.halfOrder: self.m_x - self.halfOrder]
        return img

    def propagate(self, p0, p1):
        assert(p0.shape == p1.shape == self.shape)
        lap = self.laplacian(p1)
        img = 2 * p1 - p0 + self.v2dt2 * lap
        del lap
        return img
