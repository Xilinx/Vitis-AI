import imp
import os
import sys
from torch.utils.cpp_extension import load, _import_module_from_library
from nndct_shared.utils import create_work_dir, NndctScreenLogger

_cur_dir = os.path.dirname(os.path.realpath(__file__))
_aot = False

for name in os.listdir(_cur_dir):
  if name.split(".")[-1] == "so":
    _aot = True
    break

if _aot:
  try:
    from pytorch_nndct.nn import _kernels
    nndct_kernels = _kernels
  except ImportError as e:
    NndctScreenLogger().error(f"{str(e)}")
    sys.exit(1)
  else:
    NndctScreenLogger().info(f"Loading NNDCT kernels...")
    
else:    
  if os.path.exists(os.path.join(_cur_dir, "kernel")):
    from .kernel import NN_PATH
  else:
    NN_PATH = _cur_dir
  try:
    cwd = NN_PATH
    lib_path = os.path.join(cwd, "lib")
    create_work_dir(lib_path)
    cuda_src_path = os.path.join(cwd, "../../../csrc")
    cpp_src_path = os.path.join(cwd, "src")
    source_files = []
    for name in os.listdir(cuda_src_path):
      if name.split(".")[-1] in ["cu", "cpp", "cc", "c"]:
        source_files.append(os.path.join(cuda_src_path, name))
    for name in os.listdir(cpp_src_path):
      if name.split(".")[-1] in ["cpp", "cc", "c"]:
        source_files.append(os.path.join(cpp_src_path, name))

    extra_include_paths = [
        os.path.join(cwd, "../../../include"),
        os.path.join(cwd, "include")
    ]
    nndct_kernels = load(
        name="nndct_kernels",
        sources=source_files,
        verbose=False,
        build_directory=lib_path,
        extra_include_paths=extra_include_paths)
  except ImportError as e:
    NndctScreenLogger().error(f"{str(e)}")
    sys.exit(1)
  else:
    NndctScreenLogger().info(f"Loading NNDCT kernels...")
