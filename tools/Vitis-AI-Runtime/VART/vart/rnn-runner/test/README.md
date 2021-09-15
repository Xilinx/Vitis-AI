
# Setup
```sh
cd vart
./docker_run.sh xilinx/vitis-ai-cpu:latest

cd /workspace && ./cmake.sh --cmake-options="-DENABLE_RNN_RUNNER=ON --DENABLE_CPU_RUNNER=OFF -DENABLE_SIM_RUNNER=OFF" && cd -
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/.local/Ubuntu.18.04.x86_64.Debug/lib

# U25
sudo cp /workspace/rnn-runner/xclbin/u25/dpu.xclbin /usr/lib/dpu.xclbin

# U50
sudo cp /workspace/rnn-runner/xclbin/u50lv/dpu.xclbin /usr/lib/dpu.xclbin
```

## Download model files

### Download U25 model
```sh
mkdir -p /workspace/rnn-runner/test/models
wget -O u25-2020-12-16.tar.gz https://www.xilinx.com/bin/public/openDownload?filename=u25-2020-12-16.tar.gz
tar -xvf u25-2020-12-16.tar.gz -C /workspace/rnn-runner/test/models && rm -rf u25-2020-12-16.tar.gz

cp -r <xcd>:/group/dphi_edge/vai-1.4-rnn-xmodel/ /workspace/rnn-runner/test
```

### Download Compiled Models
```sh
cp -r <xcd>:/group/dphi_edge/vai-1.4-rnn-xmodel/ /workspace/rnn-runner/test
```

## Copy test data

```sh
cd /workspace/rnn-runner/test
tar -xzvf rnn-runner-test-data.tar.gz
```

# Run tests
## Full Test
`run_test.sh` is an utility script to run 4 tests (oie-36/oie-59/customer satisfaction/imdb sentiment) in a single step. 
It provides  functionality check, latency breakup and throughput for single runner and multiple runner tests.

### Script arguments

|Argument           | Description
|-----------------  | -------
|**--help**         | Show help
| **--model-dir**   | model directory containing models for all applications
| **--device**      | target device. <br> Possible Values : **u25,  u50** <br> [default = u50]
| **--num-iters**   | number of iterations to run. <br> [default = 4]
| **--num-runners** | number of runners to use in multi-runner tests. <br> [default = 4]
| **--build-dir**   | set customized build directory. <br> [default = /home/vitis-ai-user/build/build.Ubuntu.18.04.x86_64.Release]
| **--tests-dir**   | test data directory. <br> [default = ./data]
| **--mode**        | specify execution mode. <br> Possible Values : **func, latency, throughput, all**. <br> [default = func] <br> **func** : Run functionality check alone <br> **latency** : Get latency breakup <br> **throughput** : Get application throughput <br> **all** : Run all the 3 tests.

### Example

```sh
# Run all tests for U50 with Release build
./run_test.sh \
    --model-dir=/workspace/vart/tmp/vai-1.4-rnn-xmodel/u25/xmodels-v1.0/ \
    --build-dir=/home/vitis-ai-user/build/build.Ubuntu.18.04.x86_64.Release/ \
    --mode=all --device=u50
```

## Single Tests

To run tests for a single model for debugging purposes, use the device specific test executable in the build directory

### Usage
```sh
./test_executable       \   # "test_rnn_runner_u50" or "test_rnn_runner_u25"
    <model_directory>   \   # Path to directory containing specific model
    <test_data_dir>     \   # Directory containing reference data for the model
    <number of frames>  \   # Number of frames in the input
    <number of runners> \   # Number of runners to be created 
    <number of iterations>  # Number of iterations each runner should run.
```

### Examples
```sh
# Run applications on U50 with Release build
BUILD_DIR=/home/vitis-ai-user/build/build.Ubuntu.18.04.x86_64.Release
EXECUTABLE=vart/rnn-runner/test_rnn_runner_u50

# customer satisfaction model
$BUILD_DIR/$EXECUTABLE                                                          \
    /workspace/vai-1.4-rnn-xmodel/u50/xmodels-v1.3/lstm_customer_satisfaction   \
    ./data/u50/satis                                                            \
    25 1 4

# Run imdb sentiment detection model
$BUILD_DIR/$EXECUTABLE                                                          \
    /workspace/vai-1.4-rnn-xmodel/u50/xmodels-v1.3/lstm_sentiment_detection     \
    ./data/u50/sent                                                             \
    500 1 4

# Run openie model
$BUILD_DIR/$EXECUTABLE                                                          \
    /workspace/vai-1.4-rnn-xmodel/u50/xmodels-v1.3/openie-new                   \
    ./data/u50/oie/36                                                           \
    36 1 4

# Similarly, use vart/rnn-runner/test_rnn_runner_u25 as executable for U25.
```
