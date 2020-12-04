

#
# Copyright 2019 Xilinx Inc.
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

import os
import numpy as np

def read_params(name, in_dim, out_dim, model_type):
    if isinstance(name, str):
        file_type = os.path.splitext(name)[-1]
        if file_type == '.txt':
            params = np.loadtxt(name, dtype=int)
    elif isinstance(name, np.ndarray):
        params = name
    else:
        raise ValueError("Param type must be dict or file")
    
    if model_type == 'tensorflow':
        # IGFO-->IFGO
        params = np.concatenate((params[:out_dim], params[out_dim * 2:out_dim * 3], 
                                  params[out_dim:out_dim * 2], params[out_dim * 3:]), axis=0)
    elif model_type == 'torch':
        # IFGO-->IFGO
        pass
    else:
        # IFGO-->IFGO
        pass
    return params

def read_data(name):
    if isinstance(name, str):
        file_type = os.path.splitext(name)[-1]
        if file_type == '.txt':
            data = np.loadtxt(name, dtype=int)
    elif isinstance(name, np.ndarray):
        data = name
    else:
        raise ValueError("Data type must be dict or file")
    return data
 
'''    
def read_bias():
    file_type = os.path.splitext(path)[-1]
    if file_type == '.txt':
        bias = np.loadtxt(path, dtype=float)
    
    if model_type == 'tensorflow':
        # IGFO-->IFGO
        bias = np.concatenate((bias[:out_dim], bias[out_dim * 2:out_dim * 3], 
                                  bias[out_dim:out_dim * 2], bias[out_dim * 3:]), axis=0)
    elif model_type == 'torch':
        # IFGO-->IFGO
        pass
    else:
        # IFGO-->IFGO
        pass
    return bias
'''
