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

"""vai_q_tensorflow module"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from vai_q_tensorflow.python.ops.fix_neuron_ops import fix_neuron, FixNeuron

# from utils import DecentQTransform

from vai_q_tensorflow.python import utils
from vai_q_tensorflow.python.quantize_graph import QuantizeConfig
from vai_q_tensorflow.python.quantize_graph import ConvertConstantsToVariables
from vai_q_tensorflow.python.quantize_graph import CreateOptimizedGraphDef
from vai_q_tensorflow.python.quantize_graph import CreateQuantizeTrainingGraphDef
from vai_q_tensorflow.python.quantize_graph import CreateQuantizeCalibrationGraphDef
from vai_q_tensorflow.python.quantize_graph import CreateQuantizeEvaluationGraphDef
from vai_q_tensorflow.python.quantize_graph import CreateQuantizeDeployGraphDef
from vai_q_tensorflow.python.quantize_graph import CreateQuantizeTrainingGraph
from vai_q_tensorflow.python.quantize_graph import CreateQuantizeEvaluationGraph
from vai_q_tensorflow.python.quantize_graph import ConvertFoldedBatchnorms
from vai_q_tensorflow.python.quantize_graph import CreateQuantizeDeployGraph

from vai_q_tensorflow.python.decent_q import inspect
from vai_q_tensorflow.python.decent_q import quantize_frozen
from vai_q_tensorflow.python.decent_q import quantize_train
from vai_q_tensorflow.python.decent_q import quantize_evaluate
from vai_q_tensorflow.python.decent_q import deploy_checkpoint
from vai_q_tensorflow.python.decent_q import quantize_frozen
from vai_q_tensorflow.python.decent_q import quantize_train
from vai_q_tensorflow.python.decent_q import check_float_graph
from vai_q_tensorflow.python.decent_q import convert_datatype

from vai_q_tensorflow.gen_files.version import __version__, __git_version__
