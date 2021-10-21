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
from stencil import Stencil3D
from model import Model


class RTM3D(Stencil3D):
    def __init__(self, model):
        Stencil3D.__init__(self, model)
        self.nt = model.time
        self.nxb = model.nxb
        self.nyb = model.nyb
        self.nzb = model.nzb

    def setTime(self, nt):
        self.nt = nt

    def setSrc(self, srcx, srcy, srcz, src):
        self.srcX = srcx
        self.srcY = srcy
        self.srcZ = srcz
        self.src = src

    def setB(self, nxb, nyb, nzb):
        self.nxb = nxb
        self.nyb = nyb
        self.nzb = nzb

    def setSensor(self, srcz, data):
        assert data.shape == (self.nt, self.m_x - 2 *
                              self.nxb, self.m_y - 2 * self.nyb)
        self.sensorZ = srcz
        self.sensor = data

    def computeTaper(self, fac=0.75):
        dfrac = np.sqrt(-np.log(fac)) / (self.nxb)
        self.taperx = np.zeros(self.nxb, dtype=np.float32)
        self.tapery = np.zeros(self.nyb, dtype=np.float32)
        self.taperz = np.zeros(self.nzb, dtype=np.float32)
        for i in range(self.nxb):
            self.taperx[i] = np.exp(-np.power(dfrac * (self.nxb - i), 2))
        for i in range(self.nyb):
            self.tapery[i] = np.exp(-np.power(dfrac * (self.nyb - i), 2))
        for i in range(self.nzb):
            self.taperz[i] = np.exp(-np.power(dfrac * (self.nzb - i), 2))

    def taper(self, img):
        img[:, :, :self.nzb] *= self.taperz
        img[:self.nxb, :, :self.nzb] *= self.taperx
        img[-self.nxb:, :, :self.nzb] *= np.flip(self.taperx)

        img[:, :self.nxb, :self.nzb] *= self.tapery
        img[:, -self.nxb:, :self.nzb] *= np.flip(self.tapery)

    def forward(self):
        upb = np.zeros((self.nt, self.m_x,
                        self.m_y, self.order // 2), dtype=np.float32)
        p0 = self.zeros()
        p1 = self.zeros()
        for t in range(self.nt):
            self.taper(p0)
            self.taper(p1)
            p = self.propagate(p0, p1)
            p[self.srcX, self.srcY, self.srcZ] += self.src[t]
            del p0
            p0 = p1
            p1 = p
            upb[t] = p[:, :, self.nzb - self.order // 2: self.nzb]

        return p0, p1, upb

    def backward(self, snap0, snap1, upb):
        r0 = self.zeros()
        r1 = self.zeros()
        imloc = self.zeros()
        p0 = snap1
        p1 = snap0

        for t in range(self.nt):
            self.taper(r0)
            self.taper(r1)
            r = self.propagate(r0, r1)
            r[self.nxb: self.m_x - self.nxb, self.nyb: self.m_y -
                self.nyb, self.sensorZ] += self.sensor[self.nt - 1 - t]
            del r0
            r0 = r1
            r1 = r

            if t == 0:
                p = snap1
            elif t == 1:
                p = snap0
            else:
                p = self.propagate(p0, p1)

            p[:, :, self.nzb - self.order // 2: self.nzb] = upb[self.nt - 1 - t]
            del p0
            p0 = p1
            p1 = p
            imloc += np.multiply(r1, p1)

        return imloc[self.nxb:self.m_x -
                     self.nxb, self.nyb: self.m_y -
                     self.nyb, self.nzb:self.m_z -
                     self.nzb], r0, r1, p0, p1

    def saveParams(self, path):
        Stencil3D.saveParams(self, path)
        self.taperx.tofile(os.path.join(path, "taperx.bin"))
        self.tapery.tofile(os.path.join(path, "tapery.bin"))
        self.taperz.tofile(os.path.join(path, "taperz.bin"))

        self.src.tofile(os.path.join(path, "src.bin"))
        self.v2dt2.tofile(os.path.join(path, "v2dt2.bin"))

        if hasattr(self, 'sensor'):
            self.sensor.T.tofile(os.path.join(path, "sensorT.bin"))
            self.sensor.tofile(os.path.join(path, "sensor.bin"))


def forward(args, model):
    path = args.path
    if path is None:
        path = "forward"
    if not os.path.exists(path):
        os.makedirs(path)
    stencil = RTM3D(model)

    stencil.computeTaper(model.taper_factor)

    src = stencil.ricker_wavelet(model.time, model.fpeak)
    stencil.setSrc(model.x // 2, model.y // 2, model.nzb, src)

    vpe = np.fromfile(model.vpefile, dtype=np.float32)
    stencil.setV(vpe.reshape([model.x, model.y, model.z]))

    stencil.saveParams(path)

    if args.verify != 0:
        snap0, snap1, upb = stencil.forward()
        np.transpose(snap0).tofile(os.path.join(path, "snap0.bin"))
        np.transpose(snap1).tofile(os.path.join(path, "snap1.bin"))
        for t in range(model.time):
            upb[t].tofile(os.path.join(path, "upb_%d.bin" % t))


def rtm(args, model):
    path = args.path
    if path is None:
        path = "rtm"
    if not os.path.exists(path):
        os.makedirs(path)

    stencil = RTM3D(model)

    stencil.computeTaper(model.taper_factor)

    src = stencil.ricker_wavelet(model.time, model.fpeak)
    stencil.setSrc(model.x // 2, model.y // 2, model.nzb, src)

    vpe = np.fromfile(model.vpefile, dtype=np.float32)
    stencil.setV(vpe.reshape([model.x, model.y, model.z]))

    sensor = np.random.rand(
        model.time,
        stencil.m_x -
        2 *
        stencil.nxb,
        stencil.m_y -
        2 *
        stencil.nyb).astype(
        np.float32)
    stencil.setSensor(model.nzb, sensor)

    stencil.saveParams(path)

    if args.verify != 0:
        snap0, snap1, upb = stencil.forward()
        np.transpose(snap0).tofile(os.path.join(path, "snap0.bin"))
        np.transpose(snap1).tofile(os.path.join(path, "snap1.bin"))

        imloc, r0, r1, p0, p1 = stencil.backward(snap0, snap1, upb)
        np.transpose(imloc).tofile(os.path.join(path, "imloc.bin"))
        np.transpose(p0).tofile(os.path.join(path, "p0.bin"))
        np.transpose(p1).tofile(os.path.join(path, "p1.bin"))

        np.transpose(r0).tofile(os.path.join(path, "r0.bin"))
        np.transpose(r1).tofile(os.path.join(path, "r1.bin"))


def main(args):
    if not os.path.exists(args.path):
        os.makedirs(args.path)
    model = Model(js=args.json)
    eval(args.func)(args, model)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Generate random vectors and run test.')
    parser.add_argument('--func', type=str, default='forward')
    parser.add_argument('--path', type=str, default='data')
    parser.add_argument('--json', type=str)
    parser.add_argument('--verify', action='store_true')
    args = parser.parse_args()
    if args.json is None:
        parser.print_help()
    else:
        main(args)
