# Install TensorFlow Model Optimization

It is recommended to create a Python virtual environment before proceeding to
the installation. Please see the TensorFlow installation
[guide](https://www.tensorflow.org/install/pip#2.-create-a-virtual-environment-recommended)
for more information.

### Stable Builds

To install the latest version, run the following:

```shell
# Installing with the `--upgrade` flag ensures you'll get the latest version.
pip install --user --upgrade tensorflow-model-optimization
```

For release details, see our
[release notes](https://github.com/tensorflow/model-optimization/releases).

For the required version of TensorFlow and other compatibility information, see
the API Compatibility Matrix section of the Overview page for the technique you
intend to use. For instance, for pruning, the Overview page is
[here](https://www.tensorflow.org/model_optimization/guide/pruning).

Since TensorFlow is *not* included as a dependency of the TensorFlow Model
Optimization package (in `setup.py`), you must explicitly install the TensorFlow
package (`tf-nightly` or `tf-nightly-gpu`). This allows us to maintain one
package instead of separate packages for CPU and GPU-enabled TensorFlow.

### Installing from Source

You can also install from source. This requires the
[Bazel](https://bazel.build/) build system.

```shell
# To install dependencies on Ubuntu:
# sudo apt-get install bazel git python-pip
# For other platforms, see Bazel docs above.
git clone https://github.com/tensorflow/model-optimization.git
cd model_optimization
bazel build --copt=-O3 --copt=-march=native :pip_pkg
PKGDIR=$(mktemp -d)
./bazel-bin/pip_pkg $PKGDIR
pip install --user --upgrade $PKGDIR/*.whl
```
