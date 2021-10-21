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
from rtm import RTM3D
from model import Model


def testBackward(args, model):
    path = args.path
    if path is None:
        path = "backward"
    if not os.path.exists(path):
        os.makedirs(path)
    stencil = RTM3D(model)

    stencil.setB(args.nxb, args.nyb, args.nzb)
    stencil.computeTaper(0.75)
    stencil.taperx.tofile(os.path.join(path, "taperx.bin"))
    stencil.tapery.tofile(os.path.join(path, "tapery.bin"))
    stencil.taperz.tofile(os.path.join(path, "taperz.bin"))

    stencil.computeCoef(8)
    stencil.coefx.tofile(os.path.join(path, "coefx.bin"))
    stencil.coefy.tofile(os.path.join(path, "coefy.bin"))
    stencil.coefz.tofile(os.path.join(path, "coefz.bin"))

    v2dt2 = stencil.randImg() * 1e6
    stencil.setV(v2dt2)
    stencil.v2dt2.tofile(os.path.join(path, "v2dt2.bin"))

    sensor = np.random.rand(args.time, stencil.m_x -
                            2 * stencil.nxb).astype(np.float32)
    np.transpose(sensor).tofile(os.path.join(path, "sensorT.bin"))
    sensor.tofile(os.path.join(path, "sensor.bin"))
    #np.savetxt(os.path.join(path, "sensor.txt"), sensor, delimiter=',', fmt='%.5e')
    stencil.setSensor(args.nzb, sensor)
    snap0 = stencil.randImg()
    snap1 = stencil.randImg()
    np.transpose(snap0).tofile(os.path.join(path, "snap0.bin"))
    np.transpose(snap1).tofile(os.path.join(path, "snap1.bin"))
    upb = np.random.rand(stencil.nt, args.order // 2,
                         stencil.m_x).astype(np.float32)
    upb[stencil.nt - 2, :, :] = snap0[stencil.nzb -
                                      stencil.order // 2: stencil.nzb, :]
    upb[stencil.nt - 1, :, :] = snap1[stencil.nzb -
                                      stencil.order // 2: stencil.nzb, :]
    np.transpose(upb, (0, 2, 1)).tofile(os.path.join(path, "upb.bin"))

    if args.verify != 0:
        imloc, r0, r1, p0, p1 = stencil.backward(snap0, snap1, upb)
        np.transpose(imloc).tofile(os.path.join(path, "imloc.bin"))
        np.transpose(p0).tofile(os.path.join(path, "p0.bin"))
        np.transpose(p1).tofile(os.path.join(path, "p1.bin"))
        #np.savetxt(os.path.join(path, "p0.txt"), np.transpose(p0), delimiter=',', fmt='%.5e')
        #np.savetxt(os.path.join(path, "p1.txt"), np.transpose(p1), delimiter=',', fmt='%.5e')

        np.transpose(r0).tofile(os.path.join(path, "r0.bin"))
        np.transpose(r1).tofile(os.path.join(path, "r1.bin"))
        #np.savetxt(os.path.join(path, "r0.txt"), np.transpose(r0), delimiter=',', fmt='%.5e')
        #np.savetxt(os.path.join(path, "r1.txt"), np.transpose(r1), delimiter=',', fmt='%.5e')


def testForward(args, model):
    path = args.path
    if path is None:
        path = "forward"
    if not os.path.exists(path):
        os.makedirs(path)
    stencil = RTM3D(model)
    stencil.setTime(args.time)

    stencil.setB(args.nxb, args.nyb, args.nzb)

    stencil.computeTaper(0.75)

    if args.rbc:
        stencil.taperx = np.ones(stencil.taperx.shape, dtype=np.float32)
        stencil.tapery = np.ones(stencil.tapery.shape, dtype=np.float32)
        stencil.taperz = np.ones(stencil.taperz.shape, dtype=np.float32)
    stencil.taperx.tofile(os.path.join(path, "taperx.bin"))
    stencil.tapery.tofile(os.path.join(path, "tapery.bin"))
    stencil.taperz.tofile(os.path.join(path, "taperz.bin"))

    stencil.coefx.tofile(os.path.join(path, "coefx.bin"))
    stencil.coefy.tofile(os.path.join(path, "coefy.bin"))
    stencil.coefz.tofile(os.path.join(path, "coefz.bin"))

    # src = np.ones(args.time, np.float32)
    #src = np.ones(args.time).astype(np.float32)
    src = stencil.ricker_wavelet(args.time, 15)
    src.tofile(os.path.join(path, "src_s0.bin"))
    stencil.setSrc(args.x // 2, args.y // 2, args.nzb, src)

    v2dt2 = stencil.randImg() * 1e3
    # v2dt2 = stencil.ones() * 1e3
    stencil.setV(v2dt2)
    stencil.v2dt2.tofile(os.path.join(path, "v2dt2.bin"))

    if args.verify != 0:
        snap0, snap1, upb = stencil.forward()
        snap0.tofile(os.path.join(path, "snap0.bin"))
        snap1.tofile(os.path.join(path, "snap1.bin"))
        upb.tofile(os.path.join(path, "upb.bin"))


def testRTM(args, model):
    path = args.path
    if path is None:
        path = "rtm"
    if not os.path.exists(path):
        os.makedirs(path)
    stencil = RTM3D(model)

    stencil.setB(args.nxb, args.nyb, args.nzb)
    stencil.computeTaper(0.75)
    stencil.taperx.tofile(os.path.join(path, "taperx.bin"))
    stencil.taperz.tofile(os.path.join(path, "taperz.bin"))

    stencil.computeCoef(args.order)
    stencil.coefx.tofile(os.path.join(path, "coefx.bin"))
    stencil.coefz.tofile(os.path.join(path, "coefz.bin"))

    src = np.random.rand(args.time).astype(np.float32)
    src.tofile(os.path.join(path, "src.bin"))
    #np.savetxt(os.path.join(path, "src.txt"),src,  delimiter=',', fmt='%.5e')
    stencil.setSrc(args.nzb, model[1] // 2, src)

    #v2dt2 = np.ones(model).astype(np.float32)
    v2dt2 = stencil.randImg() * 1e6
    stencil.setV(v2dt2)
    stencil.v2dt2.tofile(os.path.join(path, "v2dt2.bin"))

    sensor = np.random.rand(args.time, stencil.m_x -
                            2 * stencil.nxb).astype(np.float32)
    np.transpose(sensor).tofile(os.path.join(path, "sensorT.bin"))
    sensor.tofile(os.path.join(path, "sensor.bin"))
    stencil.setSensor(args.nzb, sensor)

    if args.verify != 0:
        snap0, snap1, upb = stencil.forward()
        np.transpose(snap0).tofile(os.path.join(path, "snap0.bin"))
        np.transpose(snap1).tofile(os.path.join(path, "snap1.bin"))

        imloc, r0, r1, p0, p1 = stencil.backward(snap0, snap1, upb)
        np.transpose(imloc).tofile(os.path.join(path, "imloc.bin"))
        np.transpose(p0).tofile(os.path.join(path, "p0.bin"))
        np.transpose(p1).tofile(os.path.join(path, "p1.bin"))
        #np.savetxt(os.path.join(path, "p0.txt"), np.transpose(p0), delimiter=',', fmt='%.5e')
        #np.savetxt(os.path.join(path, "p1.txt"), np.transpose(p1), delimiter=',', fmt='%.5e')

        np.transpose(r0).tofile(os.path.join(path, "r0.bin"))
        np.transpose(r1).tofile(os.path.join(path, "r1.bin"))
        #np.savetxt(os.path.join(path, "r0.txt"), np.transpose(r0), delimiter=',', fmt='%.5e')
        #np.savetxt(os.path.join(path, "r1.txt"), np.transpose(r1), delimiter=',', fmt='%.5e')


def testPropagate(args, model):
    path = args.path
    stencil = RTM3D(model)

    stencil.computeCoef()
    stencil.coefx.tofile(os.path.join(path, "coefx.bin"))
    stencil.coefy.tofile(os.path.join(path, "coefy.bin"))
    stencil.coefz.tofile(os.path.join(path, "coefz.bin"))

    v2 = stencil.randImg() * 1e6
    stencil.setV(v2)
    stencil.v2dt2.tofile(os.path.join(path, "v2dt2.bin"))

    mP = tuple([s // 2 for s in stencil.shape])
    p = [stencil.zeros(), stencil.zeros()]
    p[0][mP] = 1
    p[1][mP] = 1

    p[0].tofile(os.path.join(path, "ip0.bin"))
    p[1].tofile(os.path.join(path, "ip1.bin"))

    for t in range(args.time):
        prop = stencil.propagate(p[-2], p[-1])
        p.append(prop)

    p[-2].tofile(os.path.join(path, "sp0.bin"))
    p[-1].tofile(os.path.join(path, "sp1.bin"))


def testLaplacian(args, model):
    path = args.path
    stencil = Stencil3D(model)
    image = stencil.randImg() * 100
    image.tofile(os.path.join(path, "image.bin"))

    stencil.saveParams(path)

    for _ in range(args.time):
        image = stencil.laplacian(image)

    image.tofile(os.path.join(path, "result.bin"))


def main(args):
    if not os.path.exists(args.path):
        os.makedirs(args.path)
    model = Model()
    model.parse(args)
    eval(args.func)(args, model)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Generate random vectors and run test.')
    parser.add_argument('--func', type=str, default='testLaplacian')
    parser.add_argument('--path', type=str)
    parser.add_argument('--x', type=int, default=90)
    parser.add_argument('--y', type=int, default=90)
    parser.add_argument('--z', type=int, default=90)
    parser.add_argument('--nxb', type=int, default=20)
    parser.add_argument('--nyb', type=int, default=20)
    parser.add_argument('--nzb', type=int, default=20)
    parser.add_argument('--order', type=int, default=8)
    parser.add_argument('--time', type=int, default=1)
    parser.add_argument('--verify', type=int, default=1)
    parser.add_argument('--shot', type=int, default=1)
    parser.add_argument('--rbc', action="store_true")
    args = parser.parse_args()
    main(args)
