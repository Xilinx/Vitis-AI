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
import os
import math
import argparse
from operation import DataGenerator


def parse_cfg(filename):
    myvars = {}
    with open(filename) as myfile:
        for line in myfile:
            for word in line.split():
                name, var = word.split("=")
                myvars[name.strip()] = var.rstrip()
    return myvars


class fcn_dataGenerator:

    def __init__(self, dirname):
        self.dataGen = DataGenerator()
        self.dataGen.setRange(-1, 1)
        self.dataGen.setDataType(np.float32)
        self.dirname = dirname

    def setWeights(self, sizes):
        self.weights = []
        self.bias = []
        self.sizes = sizes
        self.number_of_layers = int(len(sizes) / 2)
        for i in range(self.number_of_layers):
            size_k = sizes[2 * i]
            size_n = sizes[2 * i + 1]
            self.weights.append(self.dataGen.matrix((size_k, size_n)))
            self.bias.append(self.dataGen.matrix((1, size_n)))

    def compute(
            self,
            weights_size,
            batch_size,
            number_of_models,
            number_of_inputs,
            activation):
        self.dirname = self.dirname + "_" + str(batch_size)
        os.makedirs(self.dirname, exist_ok=True)
        size_m = batch_size
        size_n = weights_size[0]
        self.inputs = []
        # generate random inputs
        for inp in range(number_of_inputs):
            input = self.dataGen.matrix((size_m, size_n))
            input.tofile(
                os.path.join(
                    self.dirname,
                    'mat_input_' +
                    str(inp) +
                    '_' +
                    str(size_m) +
                    '.bin'))
            self.inputs.append(input)
        for model in range(number_of_models):
            # generate random weights
            self.setWeights(weights_size)
            # save weights and bias to file
            for i in range(self.number_of_layers):
                self.weights[i].tofile(os.path.join(
                    self.dirname, 'matW' + str(i + 1) + '_' + str(model) + '.bin'))
                self.bias[i].tofile(os.path.join(
                    self.dirname, 'matb' + str(i + 1) + '_' + str(model) + '.bin'))
            for inp in range(number_of_inputs):
                self.C = [self.inputs[inp]]
                for i in range(self.number_of_layers):
                    local_bias = np.tile(self.bias[i], (size_m, 1))
                    c = (
                        np.matmul(
                            self.C[i],
                            self.weights[i],
                            dtype=np.float32) +
                        local_bias).astype(
                        np.float32)
                    # sigmoid
                    if i != (self.number_of_layers - 1):
                        if activation == 'tansig':
                            c = 2 / (1 + np.exp(-2 * c)) - 1
                        elif activation == 'relu':
                            c[c < 0] = 0
                        else:
                            c = 1 / (1 + np.exp(-c))
                    self.C.append(c)
                self.C[-1].tofile(os.path.join(self.dirname,
                                               'mat_sigmoid_output_input_' + str(inp) + '_model_' + str(model) + '.bin'))


def main(args):
    fcn = fcn_dataGenerator(args.dirname)
    fcn.compute(
        args.sizes,
        args.batch,
        args.model,
        args.inputs,
        args.activation)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Generate random vectors and run test.')
    parser.add_argument(
        '--sizes',
        nargs='+',
        help='sizes of weights',
        type=int,
        default=[
            356,
            30,
            30,
            20,
            20,
            3])
    parser.add_argument('--batch', type=int, default=204800, help='batch size')
    parser.add_argument(
        '--model',
        type=int,
        default=1,
        help='number of models')
    parser.add_argument(
        '--inputs',
        type=int,
        default=1,
        help='number of inputs')
    parser.add_argument(
        '--dirname',
        type=str,
        default="./data",
        help='data bin file dirname')
    parser.add_argument(
        '--activation',
        type=str,
        default='sigmoid',
        choices=['relu', 'tansig', 'sigmoid'],
        help='activations')

    args = parser.parse_args()
    main(args)