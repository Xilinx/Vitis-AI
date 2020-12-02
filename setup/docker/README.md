# vitis-ai-docker

Dockerfiles for building a unified vitis-ai release

# So what's the deal?
At a high level, we need at least two Docker containers:  
* xilinx/vitis-ai-cpu
* xilinx/vitis-ai-gpu
  
Internally they will be cached in:  
* xdock:5000/vitis-ai-cpu:X.Y.Z
* xdock:5000/vitis-ai-gpu:X.Y.Z

Where X.Y.Z are major.minor.patch version respectivly


# Build
```
docker login xdock:5000

## CPU

docker build . --buildarg VERSION=X.Y.Z -f DockerfileCPU -t  xdock:5000/vitis-ai-cpu:X.Y.Z
docker tag xdock:5000/vitis-ai-cpu:X.Y.Z xdock:5000/vitis-ai-cpu:latest
docker push xdock:5000/vitis-ai-cpu:X.Y.Z
docker push xdock:5000/vitis-ai-cpu:latest


## GPU
docker build . --buildarg VERSION=X.Y.Z -f DockerfileGPU -t  xdock:5000/vitis-ai-gpu:X.Y.Z
docker tag xdock:5000/vitis-ai-cpu:X.Y.Z xdock:5000/vitis-ai-gpu:latest
docker push xdock:5000/vitis-ai-gpu:X.Y.Z
docker push xdock:5000/vitis-ai-gpu:latest

```

# Run
By default the CPU image will be ran, use argument to override, and run GPU.  
  
Note: To have any success at all w/ GPU image, you must have GPUs, and nvidia-docker setup.  
```
to use a specific version of the docker image, set the VERSION environment variable, if not set then 'latest' will be used.
export VERSION=X.Y.Z 
./docker_run.sh                         # Run CPU Image
./docker_run.sh xdock:5000/vitis-ai:gpu:latest # Run GPU Image
```

# Sanity Test (CPU)
```
to use a specific version of the docker image, set the VERSION environment variable, if not set then 'latest' will be used.
export VERSION=X.Y.Z
./docker_run.sh

conda activate vitis-ai-caffe
python -c "import caffe"
decent_q --version # Gives version 1.2.4 which is old
/opt/vitis_ai/compression/vai_q_caffe.bin --version # Gives version 1.2.4 which is old
/opt/vitis_ai/compiler/vai_c_caffe.bin --help

conda activate vitis-ai-tensorflow
python -c "import tensorflow"
decent_q --version
/opt/vitis_ai/compression/vai_q_tensorflow.bin --version
/opt/vitis_ai/compiler/vai_c_tensorflow.bin --help
```

# Sanity Test (GPU)
*MUST HAVE GPU MACHINE*
```
export VERSION=X.Y.Z
./docker_run.sh xdock:5000/vitis-ai-gpu

conda deactivate

python -c "import caffe" # FAILS
decent_q --version # FAILS
/opt/vitis_ai/compression/vai_q_caffe.bin --version
/opt/vitis_ai/compiler/vai_c_caffe.bin # FAILS
python3 /opt/vitis_ai/compiler/vai_c_caffe.bin # FAILS

python -c "import tensorflow" # FAILS
python3 -c "import tensorflow" # SUCCEEDS
/opt/vitis_ai/compiler/vai_c_tensorflow.bin # FAILS
python3 /opt/vitis_ai/compiler/vai_c_tensorflow.bin # FAILS
/opt/vitis_ai/compression/vai_q_tensorflow.bin --version # FAILS
python3 /opt/vitis_ai/compiler/vai_q_tensorflow.bin --version # FAILS 
```

# Layered Approach to the Two Images
| Phase | CPU Dockerfile | CPU Tag | GPU Dockerfile | GPU Tag | Comments    |
|-------|----------------|---------|----------------|---------|-------------|
|  base |  ubuntu:18.04  | base-cpu| nvidia/cuda    | base-gpu| Base Image  |
|  apt  | [Dockerfile.apt](Dockerfile.apt) | apt-cpu | [Dockerfile.apt](Dockerfile.apt) | apt-gpu | System libs |
|  pip3 | [Dockerfile.pip3](Dockerfile.pip3)| pip3-cpu| [Dockerfile.pip3](Dockerfile.pip3)| pip3-gpu| Python3 System Modules |
| deephi| [Dockerfile.deephi.cpu](Dockerfile.deephi.cpu)| deephi-cpu| [Dockerfile.deephi.gpu](Dockerfile.deephi.gpu)| pip3-gpu| Installs Local Deephi Packages, CPU binaries are missing |
|  conda| [Dockerfile.conda](Dockerfile.conda)| conda-cpu| [Dockerfile.conda](Dockerfile.conda)| conda-gpu| Install MiniConda3 |
|conda create| [Dockerfile.conda.create.cpu](Dockerfile.conda.create.cpu)| conda-create-cpu| [Dockerfile.conda.create.gpu](Dockerfile.conda.create.gpu)| conda-create-gpu| Install Conda Packages for xfdnn, caffe_decent, tensorflow_decent - note they are stale |
|  xrt | [Dockerfile.xrt](Dockerfile.xrt)| pip3-cpu| [Dockerfile.xrt](Dockerfile.xrt)| xrt-gpu| Install XRT 2019.2, note this will become stale shortly after XRT release |
|  clean | [Dockerfile.clean](Dockerfile.clean)| clean-cpu| [Dockerfile.clean](Dockerfile.clean)| clean-gpu| Remove unnecessary files, aiming to reduce Docker Image size|
