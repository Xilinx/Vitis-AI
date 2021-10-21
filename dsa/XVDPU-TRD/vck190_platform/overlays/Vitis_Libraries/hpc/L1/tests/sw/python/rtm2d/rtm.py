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
from stencil import Stencil


class RTM(Stencil):
    def taper(self, img):
        for i in range(self.nzb):
            for j in range(self.nxb):
                img[i, j] *= self.taperz[i] * self.taperx[j]
            for j in range(self.nzb, self.m_x - self.nxb):
                img[i, j] *= self.taperz[i]
            for j in range(self.m_x - self.nxb, self.m_x):
                img[i, j] *= self.taperz[i] * self.taperx[self.m_x - 1 - j]

    def computeTaper(self, fac=0.75):
        dfrac = np.sqrt(-np.log(fac)) / (self.nxb)
        self.taperx = np.zeros(self.nxb, dtype=self.dtype)
        self.taperz = np.zeros(self.nzb, dtype=self.dtype)
        for i in range(self.nxb):
            self.taperx[i] = np.exp(-np.power(dfrac * (self.nxb - i), 2))
        for i in range(self.nzb):
            self.taperz[i] = np.exp(-np.power(dfrac * (self.nzb - i), 2))

    def setTaper(self, taperx, taperz):
        self.setB(taperx.size, taperz.size)
        self.taperx = taperx
        self.taperz = taperz

    def setB(self, nxb, nzb):
        self.nxb = nxb
        self.nzb = nzb

        self.nx = self.m_x - 2 * self.nxb
        self.nz = self.m_z - 2 * self.nzb

    def hyb_v(self, vbin):
        v = np.fromfile(vbin).reshape([self.nz, self.nx])

    def ricker_wavelet(self, fpeak):
        src = np.zeros(self.nt, dtype=self.dtype)
        for i in range(self.nt):
            t = i * self.dt - 1 / fpeak
            x = np.pi * t * fpeak
            x2 = x * x
            src[i] = np.exp(-x2) * (1 - 2 * x2)
        return src

    def setTime(self, nt, dt=0.001):
        self.nt = nt
        self.dt = dt

    def setSrc(self, srcz, srcx, src):
        self.srcX = srcx
        self.srcZ = srcz
        self.src = src

    def setSensor(self, srcz, data):
        assert data.shape == (self.nt, self.m_x - 2 * self.nxb)
        self.sensorZ = srcz
        self.sensor = data

    def backward(self, snap0, snap1, upb):
        r0 = self.zero()
        r1 = self.zero()
        imloc = self.zero()
        p0 = snap1
        p1 = snap0

        for t in range(self.nt):
            self.taper(r0)
            self.taper(r1)
            r = self.propagate(r0, r1)
            r[self.sensorZ, self.nxb: self.m_x -
                self.nxb] += self.sensor[- 1 - t]
            del r0
            r0 = r1
            r1 = r

            if t == 0:
                p = snap1
            elif t == 1:
                p = snap0
            else:
                p = self.propagate(p0, p1)

            p[self.nzb - self.order // 2: self.nzb, :] = upb[- 1 - t, :]
            del p0
            p0 = p1
            p1 = p
            imloc += np.multiply(r1, p1)

        return imloc[self.nzb:self.m_z - self.nzb,
                     self.nxb: self.m_x - self.nxb], r0, r1, p0, p1

    def forward(self):
        upb = np.zeros((self.nt, self.order // 2, self.m_x), dtype=self.dtype)
        p0 = self.zero()
        p1 = self.zero()
        for t in range(self.nt):
            self.taper(p0)
            self.taper(p1)
            p = self.propagate(p0, p1)
            p[self.srcZ, self.srcX] += self.src[t]
            del p0
            p0 = p1
            p1 = p
            upb[t] = p[self.nzb - self.order // 2: self.nzb]

        return p0, p1, upb


def filterImage(args, imageDim):
    coefx = np.fromfile(os.path.join(
        args.path, "filter_coefx.bin"), args.dtype)
    coefz = np.fromfile(os.path.join(
        args.path, "filter_coefz.bin"), args.dtype)

    filt = Stencil(imageDim, 4)
    filt.setCoef(coefx, coefz)
    image = np.fromfile(os.path.join(args.path, "imloc.bin"), dtype=filt.dtype)
    image = image.reshape([filt.m_x, filt.m_z]).transpose()

    filt.freelap(image).transpose().tofile(
        os.path.join(args.path, "imloc_img.bin"))


def filter(image):
    coefx = np.fromfile(os.path.join(
        args.path, "filter_coefx.bin"), args.dtype)
    coefz = np.fromfile(os.path.join(
        args.path, "filter_coefz.bin"), args.dtype)

    filt = Stencil(image.shape, 4)
    filt.setCoef(coefx, coefz)

    return filt.freelap(image)


def verifyRTM(args, imageDim):
    path = args.path
    if path is None:
        return
    stencil = RTM(imageDim, args.order, dtype=args.dtype)
    stencil.setTime(args.time)

    stencil.setB(args.nxb, args.nzb)

    stencil.taperx = np.fromfile(os.path.join(
        path, "taperx.bin"), dtype=stencil.dtype)
    stencil.taperz = np.fromfile(os.path.join(
        path, "taperz.bin"), dtype=stencil.dtype)

    stencil.coefx = np.fromfile(os.path.join(
        path, "coefx.bin"), dtype=stencil.dtype)
    stencil.coefz = np.fromfile(os.path.join(
        path, "coefz.bin"), dtype=stencil.dtype)

    if args.srcx is None:
        args.srcx = imageDim[1] // 2
    else:
        args.srcx += stencil.nxb
    if args.spx is None:
        args.spx = 1

    v2dt2 = np.fromfile(os.path.join(path, "v2dt2.bin"), dtype=stencil.dtype)
    stencil.v2dt2 = v2dt2.reshape([stencil.m_x, stencil.m_z]).transpose()

    img = None
    for i in range(args.shot):
        src = np.fromfile(os.path.join(path, "src_s%d.bin" %
                                       i), dtype=stencil.dtype)
        stencil.setSrc(args.nzb, args.srcx + i * args.spx, src)

        sensor = np.fromfile(os.path.join(
            path, "sensor_s%d.bin" % i), dtype=stencil.dtype)
        sensor = np.reshape(sensor, [args.time, stencil.m_x - 2 * stencil.nxb])

        stencil.setSensor(args.nzb, sensor)

        snap0, snap1, upb = stencil.forward()

        imloc, r0, r1, p0, p1 = stencil.backward(snap0, snap1, upb)

        if img is None:
            img = imloc
        else:
            img += imloc
        del upb
        del snap0
        del snap1
        del imloc

    imgF = filter(img)

    img = np.transpose(img)
    img.tofile(os.path.join(path, "imloc.bin"))

    img = np.reshape(img, [-1, ])
    ref = np.fromfile(os.path.join(path, "ref.bin"), stencil.dtype)

    diff = np.isclose(img, ref, 1e-3, 1e-4)
    rate = sum(diff) / img.size
    print("Match rate is %f." % rate)

    if rate < 1.0:
        adiff = np.abs(img[diff == False] - ref[diff == False])
        rdiff = np.abs((img[diff == False])/(ref[diff == False]) - 1)
        print("Max abs diff is %f at index %d." %
              (max(adiff), np.argmax(adiff)))
        print("Max rel diff is %f at index %d." %
              (max(rdiff), np.argmax(rdiff)))

    imgF = np.transpose(imgF)
    imgF.tofile(os.path.join(path, "imgF.bin"))

    imgF = np.reshape(imgF, [-1, ])
    refF = np.fromfile(os.path.join(path, "img_ns%d_nt%d.img" % (args.shot,
                                                                 args.time)), stencil.dtype)

    diff = np.isclose(imgF, refF, 1e-3, 1e-4)
    rate = sum(diff) / imgF.size

    print("Match rate is %f." % rate)

    if rate < 1.0:
        adiff = np.abs(imgF[diff == False] - refF[diff == False])
        rdiff = np.abs((imgF[diff == False])/(refF[diff == False]) - 1)
        print("Max abs diff is %f at index %d." %
              (max(adiff), np.argmax(adiff)))
        print("Max rel diff is %f at index %d." %
              (max(rdiff), np.argmax(rdiff)))


def main(args):
    if not os.path.exists(args.path):
        os.makedirs(args.path)
    if args.type == 'double':
        args.dtype = np.float64
    else:
        args.dtype = np.float32
    eval(args.func)(args, (args.depth, args.width))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Generate random vectors and run test.')
    parser.add_argument('--path', type=str)
    parser.add_argument('--func', type=str)
    parser.add_argument('--width', type=int, default=90)
    parser.add_argument('--depth', type=int, default=90)
    parser.add_argument('--nxb', type=int, default=20)
    parser.add_argument('--nzb', type=int, default=20)
    parser.add_argument('--order', type=int, default=8)
    parser.add_argument('--time', type=int, default=1)
    parser.add_argument('--verify', type=int, default=1)
    parser.add_argument('--shot', type=int, default=1)
    parser.add_argument('--type', type=str, default='float')
    parser.add_argument('--srcx', type=int)
    parser.add_argument('--spx', type=int)
    args = parser.parse_args()
    main(args)
