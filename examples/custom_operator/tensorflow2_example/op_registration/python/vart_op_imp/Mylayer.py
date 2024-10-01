'''
Copyright 2022-2023 Advanced Micro Devices Inc.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations un:der the License.
'''

import numpy as np


class Mylayer:
    def __init__(self, op):
        pass

    def calculate(self, output, input):
        if len(input) == 0:
            return

        np_input = np.array(input[0], copy=False)
        alpha_input = np.array(input[1], copy=False)
        np_output = np.asarray(output)

        alpha_data = alpha_input.reshape(-1)
        input_data = np_input.reshape(-1)
        out_data = np_output.reshape(-1)

        for i in range(np_output.size):
            if input_data[i] >= 0:
                out_data[i] = input_data[i]
            else:
                out_data[i] = input_data[i] * alpha_data[i % alpha_input.size]
