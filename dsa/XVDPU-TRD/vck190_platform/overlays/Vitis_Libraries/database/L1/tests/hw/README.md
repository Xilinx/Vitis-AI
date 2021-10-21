# Primitive Unit Tests

This folder contains HLS-based unit tests for kernel primitives.

To perform c-simulation and co-sim on all tests, run the following command in this directory:

```
make run CSIM=1 COSIM=1 XPART=xcu280-fsvh2892-2L-e-es1
```

To specify a device, either `XPART` or `DEVICE` variable can be used.

The `XPART` variable specifies directly the part, for example:

```
make COSIM=1 run XPART=xcu280-fsvh2892-2L-e-es1
```

The `DEVICE` variable specifies a platform, and the makefile will query the actualy part automatically.

```
make CSIM=1 COSIM=1 check DEVICE=u280 PLATFORM_REPO_PATHS=/my/path/to/xilinx/platforms
```

Here the `PLATFORM_REPO_PATHS` helps the tool to find the platform file (.xpfm).

**Notice that if `XPART` variable is set, `DEVICE` will be ignored.**
