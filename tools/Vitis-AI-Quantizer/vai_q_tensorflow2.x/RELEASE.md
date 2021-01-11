<!--
See https://github.com/tensorflow/model-optimization/releases/ for previous
examples of release notes. This project follows https://semver.org/.

"Tested against" references the versions of TensorFlow that TFMOT unit tests
will be run against prior to release. For 2.X, only the earliest (2.0.0) and latest (nightly)
TF releases are tested against, under the assumption that everything in between
works sufficiently well enough.
-->

# Release following 0.3.0

Keras quantization API:

 * Major Features and Improvements
   *
 * Bug Fixes and Other Changes
   * Fixed Sequential model support for BatchNorm layers that follow Conv/DepthwiseConv ([issue](https://github.com/tensorflow/model-optimization/issues/378)).
   * Improved error message for not using `quantize_scope` with custom Keras layers and objects.
 * Tested against TensorFlow nightly

Keras pruning API:

 * Tested against TensorFlow 1.14.0, 2.0.0, and nightly.


