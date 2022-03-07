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

"""decent_q module"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.contrib.decent_q.python.ops.fix_neuron_ops import fix_neuron

from tensorflow.contrib.decent_q.utils import DecentQTransform

from tensorflow.contrib.decent_q.python import utils
from tensorflow.contrib.decent_q.python.quantize_graph import QuantizeConfig
from tensorflow.contrib.decent_q.python.quantize_graph import ConvertConstantsToVariables
from tensorflow.contrib.decent_q.python.quantize_graph import CreateOptimizedGraphDef
from tensorflow.contrib.decent_q.python.quantize_graph import CreateQuantizeTrainingGraphDef
from tensorflow.contrib.decent_q.python.quantize_graph import CreateQuantizeCalibrationGraphDef
from tensorflow.contrib.decent_q.python.quantize_graph import CreateQuantizeEvaluationGraphDef
from tensorflow.contrib.decent_q.python.quantize_graph import CreateQuantizeDeployGraphDef
from tensorflow.contrib.decent_q.python.quantize_graph import CreateQuantizeTrainingGraph
from tensorflow.contrib.decent_q.python.quantize_graph import CreateQuantizeEvaluationGraph
from tensorflow.contrib.decent_q.python.quantize_graph import ConvertFoldedBatchnorms
from tensorflow.contrib.decent_q.python.quantize_graph import CreateQuantizeDeployGraph

from tensorflow.contrib.decent_q.python.decent_q import inspect
from tensorflow.contrib.decent_q.python.decent_q import quantize_frozen
from tensorflow.contrib.decent_q.python.decent_q import quantize_train
from tensorflow.contrib.decent_q.python.decent_q import quantize_evaluate
from tensorflow.contrib.decent_q.python.decent_q import deploy_checkpoint
from tensorflow.contrib.decent_q.python.decent_q import quantize_frozen
from tensorflow.contrib.decent_q.python.decent_q import quantize_train
from tensorflow.contrib.decent_q.python.decent_q import check_float_graph
