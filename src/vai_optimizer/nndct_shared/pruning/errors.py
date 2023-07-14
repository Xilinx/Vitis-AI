# MIT License
#
# Copyright (c) 2023 Advanced Micro Devices, Inc.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

class OptimizerError(Exception):
  """The base class for Vitis AI Optimizer exceptions."""

  def __init__(self, message):
    """Creates a new `OptimizerError`.

    Args:
      message: The message string describing the failure.
    """
    super().__init__()
    self._message = message

  @property
  def message(self):
    """The error message that describes the error."""
    return self._message

  def __str__(self):
    return self.message

class OptimizerDataParallelNotAllowedError(OptimizerError):
  pass

class OptimizerInvalidAnaResultError(OptimizerError):
  pass

class OptimizerInvalidArgumentError(OptimizerError):
  pass

# PyTorch
class OptimizerTorchModuleError(OptimizerError):
  pass

#Tensorflow
class OptimizerKerasModelError(OptimizerError):
  pass

class OptimizerKerasLayerError(OptimizerError):
  pass

class OptimizerNotExcludeNodeError(OptimizerError):
  pass

class OptimizerNoAnaResultsError(OptimizerError):
  pass

class OptimizerSubnetError(OptimizerError):
  pass

class OptimizerNodeError(OptimizerError):
  pass

class OptimizerUnSupportedOpError(OptimizerError):
  pass

class OptimizerInvalidNodeNameError(OptimizerError):
  pass

class OptimizerDataFormatError(OptimizerError):
  pass
