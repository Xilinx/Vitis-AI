set -e

bazel build \
  --verbose_failures \
  --config=opt \
  --cxxopt="-D_GLIBCXX_USE_CXX11_ABI=0" \
  --copt="-march=x86-64" \
  --copt="-mno-sse3" \
  --copt="-mno-sse4.1" \
  --copt="-mno-sse4.2" \
  --copt="-mno-avx" \
  --copt="-mno-avx2" \
  --copt="-mno-avx512f" \
  --copt="-mno-fma" \
  //tensorflow/tools/pip_package:build_pip_package

bazel-bin/tensorflow/tools/pip_package/build_pip_package ./tensorflow_pkg/
pip uninstall -y tensorflow
pip uninstall -y tensorflow_gpu
pip install ./tensorflow_pkg/tensorflow-1.15.2-cp36-cp36m-linux_x86_64.whl
