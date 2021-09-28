# Kernel Primitive Unit Tests

This folder contains Vivado-HLS-based unit tests for kernel primitives.

To perform c-simulation and co-sim on all tests, run the following command in this directory:

```
make CSIM=1 COSIM=1 run
```

To test device other than default `xcu200-fsgd2104-2-e`, just specify the platform name via DEVICE var, for example:

```
make CSIM=1 COSIM=1 check DEVICE=u250_xdma_201803_2 PLATFORM_REPO_PATHS=/my/path/to/xilinx/platforms
```

Here the `PLATFORM_REPO_PATHS` is used to privde locations for platform files.
