import copy as _copy
import sys as _sys

from nndct_shared.utils import NndctOption, option_util, NndctDebugLogger
from .version import __version__

__all__ = ["apis", "nn"]

    
def _init_torch_nndct_module():
  _pop_out_argv = []
  argv = _copy.copy(_sys.argv)
  for cmd_pos, option in enumerate(_sys.argv[1:], 1):
    _pop_out_argv.extend(
        option_util.add_valid_nndct_option(argv, option, cmd_pos, 'torch'))

  for item in _pop_out_argv:
    _sys.argv.remove(item)

  if NndctOption.nndct_option_list.value:
    print("Usage: python file [option]")
    print("Nndct options:")
    for option in option_util.get_all_options():
      print(option)
    _sys.exit()

  if NndctOption.nndct_help.value:
    # TODO: register api info
    pass
    # print("Nndct API Description:")
    # if "__all__" in api.__dict__:
    #   for key in api.__dict__["__all__"]:
    #     item = api.__dict__[key]
    #     if inspect.isclass(item):
    #       print(f"\nclass {key}:\n{item.__doc__}")
    #       for method_name, method in item.__dict__.items():
    #         if (not (method_name.startswith("_") or
    #                  method_name.startswith("__")) and
    #             inspect.isfunction(method) and method.__doc__ is not None):
    #           print(
    #               f"\n def {method_name}{inspect.signature(method)}:\n {method.__doc__}"
    #           )

    #     elif inspect.isfunction(item):
    #       print(f"\ndef {key}{inspect.signature(item)}:\n{item.__doc__}")
    _sys.exit()

  if NndctOption.nndct_parse_debug.value:
    option_util.set_option_value("nndct_logging_level", 1)
    NndctDebugLogger("nndct_debug.log")


_init_torch_nndct_module()

import pytorch_nndct.apis
import pytorch_nndct.nn
import pytorch_nndct.utils