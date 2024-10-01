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

class PPScatterV2:
    def __init__(self, op):
        pass

    def calculate(self, output, input):
        # print("python PPScatterV2 calculate ...")
        assert len(input) == 2
        # print("python PPScatterV2 input0", input[0])
        input_data_shape = input[0].get_tensor().dims
        assert len(input_data_shape) == 4
        # print("input shape0 ", input_data_shape )  #   [1, 64, 12000, 1]
        # print("python PPScatterV2 input1", input[1])
        input_coord_shape = input[1].get_tensor().dims
        assert len(input_coord_shape) == 3           #  [1, 12000, 4]
        # print("input shape1 ", input_coord_shape )
        output_data_shape = output.get_tensor().dims
        assert len(output_data_shape) == 4        #  [1, 64, 496, 432]
        # print("output ", output_data_shape )
        coord_numbers = input_coord_shape[1]
        assert coord_numbers == input_data_shape[2]

        batch = output_data_shape[0];
        height = output_data_shape[2];
        width = output_data_shape[3];
        channel = output_data_shape[1];
        assert input_data_shape[0] == batch
        assert channel == input_data_shape[1]

        output_idx = 0
        input_idx = 0
        x_idx = 0

        outputd = np.asarray(output)
        outputd[:] = 0

        inputd0 = np.asarray(input[0])
        inputd1 = np.asarray(input[1])
        for n in range(coord_numbers):
            x = inputd1[0][x_idx][3]
            y = inputd1[0][x_idx][2]
            # print("xy:", x, y )
            if x<0 :
                break
            for i in range(channel):
                outputd[0][i][int(y)][int(x)] = inputd0[0][i][n][0]
            x_idx+=1
