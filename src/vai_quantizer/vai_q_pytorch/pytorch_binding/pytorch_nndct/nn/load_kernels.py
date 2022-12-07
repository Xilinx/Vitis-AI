import imp
import os
import sys
import torch
from torch.utils.cpp_extension import load, _import_module_from_library
from nndct_shared.utils import create_work_dir, NndctScreenLogger, QError, QWarning

_cur_dir = os.path.dirname(os.path.realpath(__file__))
_aot = False

for name in os.listdir(_cur_dir):
  if name.split(".")[-1] == "so":
    _aot = True
    break


new_kernel = False
torch_version = torch.__version__.split('.')
if int(torch_version[0]) == 1 and int(torch_version[1]) >= 7:
  new_kernel = True

if _aot:
  try:
    if not new_kernel:
      from pytorch_nndct.nn import _kernels
      nndct_kernels = _kernels
    else:
      file_ext = ".so"
      nndct_kernel_lib = [_ for _ in os.listdir(_cur_dir) if _.endswith(file_ext)][0]
      lib_abspath = os.path.join(_cur_dir, nndct_kernel_lib)
      torch.ops.load_library(lib_abspath)
  except ImportError as e:
    NndctScreenLogger().error2user(QError.IMPORT_KERNEL, f"{str(e)}")
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
    cpu_src_path = os.path.join(cwd, "../../../csrc/cpu")
    source_files = []
    for name in os.listdir(cpu_src_path):
      if name.split(".")[-1] in ["cpp", "cc", "c"]:
        source_files.append(os.path.join(cpu_src_path, name))

    extra_include_paths = [
        os.path.join(cwd, "../../../include/cpu"),
        os.path.join(cwd, "include")
    ]

    with_cuda = False
    extra_cflags = ""
    if "CUDA_HOME" in os.environ:
      NndctScreenLogger().check2user(QError.TORCH_VERSION, f"CUDA_HOME is set in environment, \
but pytorch installed is CPU version. \
Please install CUDA version pytorch.", torch.cuda.is_available())
      cuda_src_path = os.path.join(cwd, "../../../csrc/cuda")
      for name in os.listdir(cuda_src_path):
        if name.split(".")[-1] in ["cu", "cpp", "cc", "c"]:
          source_files.append(os.path.join(cuda_src_path, name))

      cpp_src_path = os.path.join(cwd, "src/cuda")
      for name in os.listdir(cpp_src_path):
        if name.split(".")[-1] in ["cpp", "cc", "c"]:
          source_files.append(os.path.join(cpp_src_path, name))

      extra_include_paths.append(os.path.join(cwd, "../../../include/cuda"))
      with_cuda = None
    elif "ROCM_HOME" in os.environ:
      NndctScreenLogger().check2user(QError.TORCH_VERSION, f"ROCM_HOME is set in environment, \
but pytorch installed is CPU version. \
Please install CUDA version pytorch.", torch.cuda.is_available())
      hip_src_path = os.path.join(cwd, "../../../csrc/cuda")
      for name in os.listdir(hip_src_path):
        if name.split(".")[-1] in ["cu", "cpp", "cc", "c"]:
          source_files.append(os.path.join(hip_src_path, name))

      cpp_src_path = os.path.join(cwd, "src/cuda")
      for name in os.listdir(cpp_src_path):
        if name.split(".")[-1] in ["cpp", "cc", "c"]:
          source_files.append(os.path.join(cpp_src_path, name))

      extra_include_paths.append(os.path.join(cwd, "../../../include/cuda"))
      extra_cflags = os.environ.get('CPPFLAGS')
      with_cuda = None
    else:
      print("CUDA (HIP) is not available, or CUDA_HOME (ROCM_HOME) not found in the environment "
          "so building without GPU support.")
      cpp_src_path = os.path.join(cwd, "src/cpu")
      for name in os.listdir(cpp_src_path):
        if name.split(".")[-1] in ["cpp", "cc", "c"]:
          source_files.append(os.path.join(cpp_src_path, name))

    is_python_module = False if new_kernel else True
    nndct_kernels = load(
        name="nndct_kernels",
        sources=source_files,
        verbose=False,
        build_directory=lib_path,
        extra_cflags=[extra_cflags],
        extra_include_paths=extra_include_paths,
        with_cuda=with_cuda,
        is_python_module=is_python_module)

  except ImportError as e:
    NndctScreenLogger().error2user(QError.IMPORT_KERNEL, f"{str(e)}")
    sys.exit(1)
  else:
    NndctScreenLogger().info(f"Loading NNDCT kernels...")
