# Multi-tenant and Multi-FPGA deployment modes

## Introduction

In the cloud, Vitis-AI uses an FPGA resource manager nicknamed `Butler`. This automatically manages the FPGAs available on a single system. It also enables easy sharing of FPGAs on the system over multiple simultaneous users. The following is a description of common user modes enabled by our FPGA resource manager.

## Single FPGA, Single Vitis-AI Runner

The Vitis-AI Runner provides a simple software API to submit jobs to Vitis-AI DPU accelerators and collect results. This is a universal API that works for both Cloud and Edge, for any Vitis-AI accelerator.

Example:
```c++
job_id = runner.execute_async(...);
runner.wait(job_id);
```

After a Vitis-AI runner object is created, an application may share the runner object to service multiple users. E.g., each user may make an independent call to `runner.execute_async()`, which returns a `job_id`. Each user can independently wait for their job to finish by calling `runner.wait(job_id)`.

## Single FPGA, Multiple Vitis-AI Runners

Multiple Vitis-AI Runners may timeshare the same FPGA. The FPGA resource manager coordinates access to ensure each user gets a turn. If the FPGA is busy, the process is made to wait until the FPGA is free.

Example (on a system with 1 FPGA):
```sh
cd examples/deployment_modes

## On terminal 1:
./run.sh -t test_classify
# (process runs immediately because FPGA is free)

## On terminal 2: 
./run.sh -t test_classify
# (process waits because terminal 1 is using FPGA, then runs as soon as FPGA is free)
```

## Multiple Users, Multiple FPGAs

A system may have multiple FPGA boards installed on the system. E.g., a combination of Alveo U250s and U200s. 

An application process requests for an FPGA by passing a directory of available ML xclbins (E.g., DPU_u250.xclbin, DPU_u200.xclbin). The FPGA resource manager then automatically finds an available FPGA matching one of the provided ML xclbins. 

Example (on a system with 1 U250 and 1 U200):
```sh
cd examples/deployment_modes

## In terminal 1:
./run.sh -t test_classify
# (process automatically gets assigned to the U250 and runs immediately)

## In terminal 2:
./run.sh -t test_classify
# (process automatically gets assigned to the U200 and runs immediately)
```

The FPGA resource manager also provides an API for advanced users to provide User-defined Functions to control how FPGAs are selected.

## Single FPGA, Multiple Models

In the previous examples, we assume that each Vitis runner acquires the entire FPGA and uses the whole FPGA for one ML model. 

However, the FPGA resource manager also supports CU-level (or SLR-level) allocation. I.e., instead of deploying the same ML model to the entire FPGA, the user may request a single CU, and deploy a different ML model to different SLRs in the FPGA. 

E.g., there are 4 SLRs on an Alveo U250. This allows up to 4 different ML models to be deployed. E.g., Resnet50 on SLR0, Inception_v3 on SLR1, Yolo_v2 on SL2 and SSD on SLR3.

Example:
```sh
cd examples/deployment_modes
./run.sh -t multinet
```
This example deploys Resnet50 on one SLR and Inception_v1 on another SLR. Two Vitis Runners are created (one for each ML model). The Vitis Runners can be used concurrently (E.g. different threads, different processes or different users), as the CU in each SLR is independent.

## Single User, Multiple FPGAs

User can acquire multiple FPGAs using Single Vitis-AI runner. [AI Kernel Scheduler](../apps/aks/README.md) provides command line options to input number of FPGAs in a system. 

Example:
```sh
## Run ResNet50 on multiple FPGAs
cd apps/aks
./aks.sh -m resnet50 -d1 <image-dir> -n <num-of-fpgas>
```

## Additional Examples

See [Neptune](../neptune/README.md) for an FPGA microservice server. This deploys a different ML model on different SLRs on one FPGA .

See [AI Kernel Scheduler](../apps/aks/README.md) for an Multi-Graph, Multi-FPGA end-to-end C++ executor.