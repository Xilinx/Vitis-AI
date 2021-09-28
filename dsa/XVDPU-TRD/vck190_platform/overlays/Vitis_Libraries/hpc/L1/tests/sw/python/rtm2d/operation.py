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
from rtm import RTM


def testBackward(args, imageDim):
    path = args.path
    if path is None:
        path = "backward"
    if not os.path.exists(path):
        os.makedirs(path)
    stencil = RTM(imageDim, args.order, dtype=args.dtype)
    stencil.setTime(args.time)

    stencil.setB(args.nxb, args.nzb)
    stencil.computeTaper(0.75)
    stencil.taperx.tofile(os.path.join(path, "taperx.bin"))
    stencil.taperz.tofile(os.path.join(path, "taperz.bin"))

    stencil.coefx.tofile(os.path.join(path, "coefx.bin"))
    stencil.coefz.tofile(os.path.join(path, "coefz.bin"))

    v2 = stencil.randImg() * 1e6
    # v2 = np.ones(imageDim).astype(stencil.dtype) * 1e6
    stencil.setV(v2)
    np.transpose(stencil.v2dt2).tofile(os.path.join(path, "v2dt2.bin"))

    sensor = np.random.rand(args.time, stencil.m_x -
                            2 * stencil.nxb).astype(stencil.dtype)
    np.transpose(sensor).tofile(os.path.join(path, "sensorT.bin"))
    sensor.tofile(os.path.join(path, "sensor_s0.bin"))
    sensor.tofile(os.path.join(path, "sensor.bin"))
    #np.savetxt(os.path.join(path, "sensor.txt"), sensor, delimiter=',', fmt='%.5e')
    stencil.setSensor(args.nzb, sensor)
    snap0 = stencil.randImg()
    snap1 = stencil.randImg()
    np.transpose(snap0).tofile(os.path.join(path, "snap0.bin"))
    np.transpose(snap1).tofile(os.path.join(path, "snap1.bin"))
    upb = np.random.rand(stencil.nt, args.order // 2,
                         stencil.m_x).astype(stencil.dtype)
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


def testForward(args, imageDim):
    path = args.path
    if path is None:
        path = "forward"
    if not os.path.exists(path):
        os.makedirs(path)
    stencil = RTM(imageDim, args.order, dtype=args.dtype)
    stencil.setTime(args.time)

    stencil.setB(args.nxb, args.nzb)
    stencil.computeTaper(0.75)
    stencil.taperx.tofile(os.path.join(path, "taperx.bin"))
    stencil.taperz.tofile(os.path.join(path, "taperz.bin"))

    stencil.coefx.tofile(os.path.join(path, "coefx.bin"))
    stencil.coefz.tofile(os.path.join(path, "coefz.bin"))

    src = np.random.rand(args.time).astype(stencil.dtype)
    # src = stencil.ricker_wavelet(15)
    src.tofile(os.path.join(path, "src_s0.bin"))
    src.tofile(os.path.join(path, "src.bin"))
    #np.savetxt(os.path.join(path, "src.txt"),src,  delimiter=',', fmt='%.5e')
    stencil.setSrc(args.nzb, imageDim[1] // 2, src)

    # v2dt2 = np.ones(imageDim).astype(stencil.dtype)
    v2dt2 = stencil.randImg() * 1e6
    stencil.setV(v2dt2)
    np.transpose(stencil.v2dt2).tofile(os.path.join(path, "v2dt2.bin"))

    if args.verify != 0:
        snap0, snap1, upb = stencil.forward()
        np.transpose(snap0).tofile(os.path.join(path, "snap0.bin"))
        np.transpose(snap1).tofile(os.path.join(path, "snap1.bin"))
        np.transpose(upb, (0, 2, 1)).tofile(os.path.join(path, "upb.bin"))

        #np.savetxt(os.path.join(path, "snap0.txt"), np.transpose(snap0), delimiter=',', fmt='%.5e')
        #np.savetxt(os.path.join(path, "snap1.txt"), np.transpose(snap1), delimiter=',', fmt='%.5e')


def verifyRTM(args, imageDim):
    path = args.path
    if path is None:
        return
    stencil = RTM(imageDim, args.order, dtype=args.dtype)
    stencil.setTime(args.time)

    stencil.setB(args.nxb, args.nzb)

    stencil.taperx = np.fromfile(os.path.join(
        path, "taperx.bin"), dtype=np.float32)
    stencil.taperz = np.fromfile(os.path.join(
        path, "taperz.bin"), dtype=np.float32)

    stencil.coefx = np.fromfile(os.path.join(
        path, "coefx.bin"), dtype=np.float32)
    stencil.coefz = np.fromfile(os.path.join(
        path, "coefz.bin"), dtype=np.float32)

    if args.srcx is None:
        args.srcx = imageDim[1] // 2
    else:
        args.srcx += stencil.nxb

    src = np.fromfile(os.path.join(path, "src.bin"), dtype=np.float32)
    stencil.setSrc(args.nzb, args.srcx, src)

    v2dt2 = np.fromfile(os.path.join(path, "v2dt2.bin"), dtype=np.float32)
    stencil.v2dt2 = v2dt2.reshape([stencil.m_x, stencil.m_z]).transpose()

    img = None
    for i in range(args.shot):
        sensor = np.fromfile(os.path.join(
            path, "sensor_s%d.bin" % i), dtype=np.float32)
        sensor = np.reshape(sensor, [args.time, stencil.m_x - 2 * stencil.nxb])

        stencil.setSensor(args.nzb, sensor)

        snap0, snap1, upb = stencil.forward()

        refP = np.fromfile(os.path.join(path, "P.bin"), np.float32).reshape(
            [stencil.m_x, stencil.m_z]).transpose()
        refPP = np.fromfile(os.path.join(path, "PP.bin"), np.float32).reshape(
            [stencil.m_x, stencil.m_z]).transpose()

        imloc, r0, r1, p0, p1 = stencil.backward(snap0, snap1, upb)

        if img is None:
            img = imloc
        else:
            img += imloc
        del upb
        del snap0
        del snap1
        del imloc

    img = np.transpose(img)
    img.tofile(os.path.join(path, "imloc.bin"))

    img = np.reshape(img, [-1, ])
    ref = np.fromfile(os.path.join(path, "ref.bin"), np.float32)

    diff = np.isclose(img, ref, 1e-3, 1e-4)
    print(sum(diff) / img.size)


def testRTM(args, imageDim):
    path = args.path
    if path is None:
        path = "rtm"
    if not os.path.exists(path):
        os.makedirs(path)
    stencil = RTM(imageDim, args.order, dtype=args.dtype)
    stencil.setTime(args.time)

    stencil.setB(args.nxb, args.nzb)
    stencil.computeTaper(0.75)
    stencil.taperx.tofile(os.path.join(path, "taperx.bin"))
    stencil.taperz.tofile(os.path.join(path, "taperz.bin"))

    stencil.coefx.tofile(os.path.join(path, "coefx.bin"))
    stencil.coefz.tofile(os.path.join(path, "coefz.bin"))

    # src = [stencil.ricker_wavelet(15) for i in range(args.shot)]
    src = np.random.rand(args.shot, args.time).astype(stencil.dtype)
    for i in range(args.shot):
        src[i].tofile(os.path.join(path, "src_s%d.bin" % i))
    #np.savetxt(os.path.join(path, "src.txt"),src,  delimiter=',', fmt='%.5e')

    #v2dt2 = np.ones(imageDim).astype(stencil.dtype)
    v2dt2 = stencil.randImg() * 1e6
    stencil.setV(v2dt2)
    np.transpose(stencil.v2dt2).tofile(os.path.join(path, "v2dt2.bin"))

    sensor = np.random.rand(args.shot, args.time, stencil.m_x -
                            2 * stencil.nxb).astype(stencil.dtype)
    for i in range(args.shot):
        sensor[i].tofile(os.path.join(path, "sensor_s%d.bin" % i))

    if args.verify != 0:
        if not args.fsx:
            args.fsx = args.width // 2

        img = None
        for i in range(args.shot):

            stencil.setSrc(args.nzb, args.fsx + i * args.sp, src[i])
            stencil.setSensor(args.nzb, sensor[i])

            snap0, snap1, upb = stencil.forward()
            np.transpose(snap0).tofile(os.path.join(path, "snap0_s%d.bin" % i))
            np.transpose(snap1).tofile(os.path.join(path, "snap1_s%d.bin" % i))

            imloc, r0, r1, p0, p1 = stencil.backward(snap0, snap1, upb)

            if img is None:
                img = imloc
            else:
                img += imloc
            np.transpose(imloc).tofile(os.path.join(path, "imloc_s%d.bin" % i))
            np.transpose(p0).tofile(os.path.join(path, "p0_s%d.bin" % i))
            np.transpose(p1).tofile(os.path.join(path, "p1_s%d.bin" % i))
            #np.savetxt(os.path.join(path, "p0.txt"), np.transpose(p0), delimiter=',', fmt='%.5e')
            #np.savetxt(os.path.join(path, "p1.txt"), np.transpose(p1), delimiter=',', fmt='%.5e')

            np.transpose(r0).tofile(os.path.join(path, "r0_s%d.bin" % i))
            np.transpose(r1).tofile(os.path.join(path, "r1_s%d.bin" % i))
            #np.savetxt(os.path.join(path, "r0.txt"), np.transpose(r0), delimiter=',', fmt='%.5e')
            #np.savetxt(os.path.join(path, "r1.txt"), np.transpose(r1), delimiter=',', fmt='%.5e')
            del upb
            del snap0
            del snap1
            del imloc

        np.transpose(img).tofile(os.path.join(path, "imloc.bin"))
        # np.savetxt(os.path.join(path, "image_out.txt"),
        #           img, delimiter=',', fmt='%.5e')


def testLaplacian(args, imageDim):
    path = args.path
    stencil = RTM(imageDim, args.order, dtype=args.dtype)
    image = stencil.randImg() * 100
    np.transpose(image).tofile(os.path.join(path, "image.bin"))
    np.savetxt(os.path.join(path, "img_in.txt"),
               image, delimiter=',', fmt='%.5e')
    stencil.coefx.tofile(os.path.join(path, "coefx.bin"))
    stencil.coefx.tofile(os.path.join(path, "coefz.bin"))

    for _ in range(args.time):
        image = stencil.laplacian(image)

    np.transpose(image).tofile(os.path.join(path, "result.bin"))
    # np.savetxt(os.path.join(path, "img_out.txt"),
    #           image, delimiter=',', fmt='%.5e')


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
    parser.add_argument('--func', type=str, default='testForward')
    parser.add_argument('--path', type=str)
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
    parser.add_argument('--fsx', type=int)
    parser.add_argument('--sp', type=int, default=0)
    args = parser.parse_args()
    main(args)
