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
from matplotlib import pyplot as plt
from matplotlib import animation


class Stencil3D:
    def __init__(self, model):
        self.model = model
        self.shape = (model.x, model.y, model.z)
        self.m_x = model.x
        self.m_y = model.y
        self.m_z = model.z

        self.dx = model.dx
        self.dy = model.dy
        self.dz = model.dz

        self.dt = model.dt

        self.order = model.order

        self.halfOrder = self.order // 2

        self.computeCoef()

    def randImg(self):
        img = np.random.rand(self.m_x, self.m_y, self.m_z).astype(np.float32)
        img[0:self.halfOrder, :, :] = 0
        img[self.m_x - self.halfOrder:, :, :] = 0
        img[:, 0:self.halfOrder, :] = 0
        img[:, self.m_y - self.halfOrder:, :] = 0
        img[:, :, 0:self.halfOrder] = 0
        img[:, :, self.m_z - self.halfOrder:] = 0
        return img

    def ricker_wavelet(self, nt, fpeak):
        wp = fpeak * np.pi
        src = np.zeros(nt, dtype=np.float32)
        for i in range(nt):
            t = i * self.dt - 1 / fpeak
            x = wp * t
            x2 = x ** 2
            src[i] = np.exp(-x2) * (1 - 2 * x2)
        return src

    def setV(self, v):
        self.v2dt2 = (v * self.dt) ** 2

    def computeCoef(self):
        coef = np.zeros(self.order + 1, dtype=np.float32)
        self.coefx = np.zeros(self.order + 1, dtype=np.float32)
        self.coefy = np.zeros(self.order + 1, dtype=np.float32)
        self.coefz = np.zeros(self.order + 1, dtype=np.float32)
        if self.order == 2:
            coef[0] = 1
            coef[1] = -2
            coef[2] = 1
        elif self.order == 4:
            coef[0] = -1 / 12
            coef[1] = 4 / 3
            coef[2] = -5 / 2
            coef[3] = 4 / 3
            coef[4] = -1 / 12
        elif self.order == 6:
            coef[0] = 1 / 90
            coef[1] = -3 / 20
            coef[2] = 3 / 2
            coef[3] = -49 / 18
            coef[4] = 3 / 2
            coef[5] = -3 / 20
            coef[6] = 1 / 90
        elif self.order == 8:
            coef[0] = -1 / 560
            coef[1] = 8 / 315
            coef[2] = -1 / 5
            coef[3] = 8 / 5
            coef[4] = -205 / 72
            coef[5] = 8 / 5
            coef[6] = -1 / 5
            coef[7] = 8 / 315
            coef[8] = -1 / 560
        else:
            print("Order is not supported.")

        for i in range(self.order + 1):
            self.coefz[i] = coef[i] / self.dz / self.dz
            self.coefy[i] = coef[i] / self.dy / self.dy
            self.coefx[i] = coef[i] / self.dx / self.dx

    def ones(self):
        return np.ones(self.shape, dtype=np.float32)

    def zeros(self):
        return np.zeros(self.shape, dtype=np.float32)

    def laplacian(self, image):
        assert(image.shape == self.shape)
        lap = self.zeros()
        for o in range(1 + self.order):
            lap[self.halfOrder:self.m_x - self.halfOrder,
                self.halfOrder:self.m_y - self.halfOrder,
                self.halfOrder:self.m_z - self.halfOrder] += self.coefx[o] * image[o: self.m_x - self.order + o,
                                                                                   self.halfOrder:self.m_y - self.halfOrder,
                                                                                   self.halfOrder:self.m_z - self.halfOrder] + self.coefy[o] * image[self.halfOrder: self.m_x - self.halfOrder,
                                                                                                                                                     o:self.m_y - self.order + o,
                                                                                                                                                     self.halfOrder:self.m_z - self.halfOrder] + self.coefz[o] * image[self.halfOrder: self.m_x - self.halfOrder,
                                                                                                                                                                                                                       self.halfOrder:self.m_y - self.halfOrder,
                                                                                                                                                                                                                       o:self.m_z - self.order + o]
        # for i in range(self.halfOrder, self.m_x - self.halfOrder):
        #    for j in range(self.halfOrder, self.m_y - self.halfOrder):
        #        for k in range(self.halfOrder, self.m_z - self.halfOrder):
        #            delta = 0
        #            for o in range(1 + self.order):
        #                deltax = self.coefx[o] * \
        #                    image[i - self.order // 2 + o][j][k]
        #                deltay = self.coefy[o] * \
        #                    image[i][j - self.order // 2 + o][k]
        #                deltaz = self.coefz[o] * \
        #                    image[i][j][k - self.order // 2 + o]
        #                delta += deltax + deltay + deltaz
        #            lap[i][j][k] += delta

        return lap

    def propagate(self, p0, p1):
        assert(p0.shape == p1.shape == self.shape)
        lap = self.laplacian(p1)
        img = 2 * p1 - p0 + self.v2dt2 * lap
        del lap
        return img

    def saveParams(self, path):

        self.coefx.tofile(os.path.join(path, "coefx.bin"))
        self.coefy.tofile(os.path.join(path, "coefy.bin"))
        self.coefz.tofile(os.path.join(path, "coefz.bin"))


def testWaveProp(args, v2, shape, srcP):
    st = Stencil3D(shape, args.order)
    st.setV(v2)
    src = st.ricker_wavelet(args.time)
    p = [st.zeros(), st.zeros()]
    #p[0][srcP] = 1
    #p[1][srcP] = 1
    for t in range(args.time):
        prop = st.propagate(p[-2], p[-1])
        prop[srcP] += src[t]
    #    pdb.set_trace()
        p.append(prop)

    return p[2:]


def energyExam(fs):
    for i, f in enumerate(fs):
        energy = np.linalg.norm(f.reshape([-1, 1]))
        print(i, energy)


def visualize(fs, coo):
    fms = len(fs)
    fig = plt.figure()
    l = fs[0][coo].size
    x = range(l)
    ax = plt.axes(xlim=(0, l), ylim=(-0.05, 0.05))
    line, = ax.plot([], [], lw=2)

    def init():
        line.set_data([], [])
        return line,

    def data(i):
        line.set_data(x, fs[i][coo])
        return line,
    anim = animation.FuncAnimation(fig, data, init_func=init,
                                   frames=fms, interval=1, blit=True)
    anim.save('basic_animation.mp4', fps=30, extra_args=['-vcodec', 'libx264'])

    plt.show()


def main(args):
    shape = (args.x, args.y, args.z)

    assert args.x > args.order
    assert args.y > args.order
    assert args.z > args.order

    srcP = tuple([s // 2 for s in shape])
    v2 = 1e6 * np.ones(shape, dtype=np.float32)
    #src = np.zeros(args.time, dtype=np.float32)
    #src[1] = 1
    fields = testWaveProp(args, v2, shape, srcP)
#    fields.tofile("fdtd.bin")
    # energyExam(fields)
    visualize(fields, tuple([range(args.x), srcP[1], srcP[2]]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Generate random vectors and run test.')
    parser.add_argument('--path', type=str)
    parser.add_argument('--x', type=int, default=1009)
    parser.add_argument('--y', type=int, default=9)
    parser.add_argument('--z', type=int, default=9)
    parser.add_argument('--nxb', type=int, default=10)
    parser.add_argument('--nyb', type=int, default=10)
    parser.add_argument('--nzb', type=int, default=10)
    parser.add_argument('--order', type=int, default=8)
    parser.add_argument('--time', type=int, default=10000)
    args = parser.parse_args()
    main(args)
