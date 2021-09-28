#!/usr/bin/env python3
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

import numpy as np
import sys
import os
import argparse
import pdb


def main(args):
    if args.datatype == 'float':
        dtype = np.float32
    elif args.datatype == 'double':
        dtype = np.float64
    else:
        sys.exit("Wrong data type received.")

    A = np.random.random((args.p_m, args.p_n)).astype(dtype=dtype)
    x = np.random.random(args.p_n).astype(dtype=dtype)
    b = A @ x

    A.tofile(os.path.join(args.path, 'A.mat'))
    x.tofile(os.path.join(args.path, 'x.mat'))
    b.tofile(os.path.join(args.path, 'b.mat'))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Generate SPD Matrix.')
    parser.add_argument(
        '--path',
        type=str,
        required=True,
        help='result file')
    parser.add_argument(
        '--p_m',
        type=int,
        default=32,
        help='Dense matrix row')
    parser.add_argument(
        '--p_n',
        type=int,
        default=32,
        help='Dense matrix col')
    parser.add_argument(
        '--datatype',
        type=str,
        default='double',
        help="data type"
    )
    args = parser.parse_args()
    main(args)
