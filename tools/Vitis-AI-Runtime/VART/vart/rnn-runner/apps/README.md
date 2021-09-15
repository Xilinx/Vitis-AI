# Testing Application Models

1. Log into a public xilinx vitis-ai docker
    ```sh
    docker pull xilinx/vitis-ai-cpu:latest
    ./docker_run.sh xilinx/vitis-ai-cpu:latest
    ```

1. Create RNN conda environments
    ```sh
    cd /workspace
    git clone http://xcdl190260/aisw/vart.git
    conda env create -f /workspace/vart/rnn-runner/apps/rnn_tf2.yml
    conda env create -f /workspace/vart/rnn-runner/apps/rnn_pytorch1p1.yml
    ```

1. Copy xclbin
    ```sh
    # [For U25]
    sudo cp /workspace/vart/rnn-runner/xclbin/u25/dpu.xclbin /usr/lib/dpu.xclbin

    # [For U50LV]
    sudo cp /workspace/vart/rnn-runner/xclbin/u50lv/dpu.xclbin /usr/lib/dpu.xclbin
    ```

1. Compile latest VART with Python support in any RNN conda environment
    ```sh
    conda activate rnn_tf_2.0

    # Get rid of other conda envs from PATH
    export PATH=`echo $PATH | tr ":" "\n" | grep -v "vitis_ai\/conda" | tr "\n" ":"`

    # Compile XIR
    cd /workspace/tools/Vitis-AI-Runtime/VART/xir
    ./cmake.sh --clean --build-only --build-python
    cp ~/build/build.Ubuntu.18.04.x86_64.Debug/xir/src/python/xir.cpython-37m-x86_64-linux-gnu.so $CONDA_PREFIX/lib/python3.7/site-packages/
    cp ~/build/build.Ubuntu.18.04.x86_64.Debug/xir/src/python/xir.cpython-37m-x86_64-linux-gnu.so $CONDA_PREFIX/../rnn_pytorch_1.1/lib/python3.7/site-packages/

    # Compile VART
    cd /workspace/vart
    ./cmake.sh --cmake-options="-DENABLE_RNN_RUNNER=ON -DENABLE_CPU_RUNNER=OFF -DENABLE_SIM_RUNNER=OFF" --build-only --build-python --clean
    sudo cp ~/build/build.Ubuntu.18.04.x86_64.Debug/vart/runner/libvart-runner.so* /usr/lib/
    sudo cp ~/build/build.Ubuntu.18.04.x86_64.Debug/vart/rnn-runner/libvart-rnn-runner.so* /usr/lib/
    cp /home/vitis-ai-user/build/build.Ubuntu.18.04.x86_64.Debug/vart/runner/vart.cpython-37m-x86_64-linux-gnu.so $CONDA_PREFIX/lib/python3.7/site-packages/
    cp /home/vitis-ai-user/build/build.Ubuntu.18.04.x86_64.Debug/vart/runner/vart.cpython-37m-x86_64-linux-gnu.so $CONDA_PREFIX/../rnn_pytorch_1.1/lib/python3.7/site-packages/
    conda deactivate
    ```

1. Get the compiled model files for your device

    ```sh
    cp -r <XCD>:/group/dphi_edge/vai-1.4-rnn-xmodel /workspace/vart/rnn-runner/models
    ```

1. Now follow the instructions in each app's directory
    1. [Customer Satisfaction](./customer_satisfaction)
    1. [IMDB Sentiment Analysis](./imdb_sentiment_detection)
    1. [OpenIE](./open_information_extraction)


#### License
Copyright 2021 Xilinx Inc.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
