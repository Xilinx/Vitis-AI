from .torch_script_writer import TorchQuantScriptWriter, TorchScriptWriter
from typing import Callable


def get_script_writer(enable_quant: bool) -> Callable:
  return TorchQuantScriptWriter() if enable_quant else TorchScriptWriter()
  