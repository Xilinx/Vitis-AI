#### DPU for RNN v1.0-beta
##### Introduction

The Xilinx Deep Learning Processing Unit (DPU) for Recurrent Neural Network is a customized inference engine optimized for recurrent nueral networks. It is designed to implement and accelerate most variations of recurrent neural network, such as standard RNN, Gate Recurrent Unit (GRU), Unidirectional and Bidirectional Long Short-Term Memory (LSTM). 

DPU for RNN v1.0-beta release includes two xclbin files which can be programed into Alveo U25 and Alveo U50LV cards, customized runner in the VART, three example LSTM applications, as well as LSTM quantizer and compiler tools. Please check the supported operation types and constraints from the [tools](../../tools/RNN) first for new target networks.
- Toolchain - [RNN Quantizer](../../tools/RNN/rnn_quantizer)
- Toolchain - [RNN Compiler](../../tools/RNN/rnn_compiler)

The DPU for RNN features are:
- XCLBIN files
    - Alveo U25: DPURADR16L-1.0.0
    - Avleo U50LV: DPURAHR16L-1.0.0
- Architecture
    - One AXI-Lite slave interface for accessing configuration and status registers.
    - One AXI master interface for accessing instructions.
    - Several AXI master interfaces for accessing model parameters and input.
    - The kernel batch size on Alveo U25 and U50LV are 1 and 7.
- Supported operation types
    - Matrix-vector multiplication
    - Element-wise multiplication
    - Element-wise addition
    - Sigmoid
    - Tanh
##### DPU for RNN directory structure introduction

    dpu-for-rnn
    |---- README.md
    |---- docker_run.sh
    |---- xclbin
    |     |---- u25
    |     |     |---- dpu.xclbin
    |     |---- u50lv
    |           |---- dpu.xclbin
    |---- scripts
    |     |---- install.sh
    |     |---- setup_u25.sh
    |     |---- setup_u50lv.sh
    |---- app
    |     |---- customer_satisfaction
    |     |     |---- README
    |     |     |---- setup.sh
    |     |     |---- build_libdpu4rnn.sh
    |     |     |---- run_cpu_e2e.py 
    |     |     |---- run_cpu_e2e_batch1.py 
    |     |     |---- run_dpu_e2e.py 
    |     |     |---- backup
    |     |     |     |---- hdf5_format.py
    |     |---- imdb_sentiment_detection
    |     |     |---- README
    |     |     |---- setup.sh
    |     |     |---- build_libdpu4rnn.sh
    |     |     |---- demo.csv 
    |     |     |---- run_cpu_e2e.py 
    |     |     |---- run_dpu_e2e.py 
    |     |     |---- backup
    |     |     |     |---- hdf5_format.py
    |     |---- open_information_extraction
    |     |     |---- README
    |     |     |---- setup.sh
    |     |     |---- build_libdpu4rnn.sh
    |     |     |---- run_cpu.sh 
    |     |     |---- run_cpu_one_trans.sh 
    |     |     |---- run_dpu.sh 
    |     |     |---- run_dpu_one_trans.sh 
    |     |     |---- test_device_close.sh 
    |     |     |---- backup
    |     |     |     |---- benchmark.py
    |     |     |     |---- moveConf.py
    |     |     |     |---- run_oie.py
    |     |     |     |---- run_oie_test.py
    |     |     |     |---- stacked_alternating_lstm_cpu.py
    |     |     |     |---- stacked_alternating_lstm_dpu.py
    |     |     |     |---- stacked_alternating_lstm_dpu_new.py
    |     |     |     |---- tabReader.py
    |     |     |---- test 
    |     |     |     |---- test.oie.sent
    |     |     |     |---- test_in.txt
    |     |     |---- weights 
    |     |     |     |---- config.json
    |     |     |     |---- vocabulary 
    |     |     |     |     |---- labels.txt
    |     |     |     |     |---- non_padded_namespaces.txt
    |     |     |     |     |---- tokens.txt
    |---- libdpu4rnn
    |     |---- CMakeLists.txt
    |     |---- make.sh 
    |     |---- test.cpp
    |     |---- test.py
    |     |---- include
    |     |     |---- dpu4rnn.hpp
    |     |---- python
    |     |     |---- dpu4rnn_python.cpp
    |     |---- src
    |     |     |---- dpu4rnn.cpp
    |     |     |---- dpu4rnn_img.cpp
    |     |     |---- dpu4rnn_imp.hpp

##### Quick Start from examples
1. Install Platform.
    
    Alveo U25 Platform:
    ```
    xilinx-cmc-u25 2.1.2-2955241
    xilinx-u25-gen3x8-xdma-base 1-2953517
    xilinx-u25-gen3x8-xdma-validate 1-2954712
    ```
    Alveo U50LV Platform:
    ```
    xilinx-cmc-u50-1.0.20-2853996.noarch
    xilinx-u50lv-gen3x4-xdma-base-2-2902115.noarch
    xilinx-u50lv-gen3x4-xdma-validate-2-2902115.noarch
    xilinx-sc-fw-u50-5.0.27-2.e289be9.noarch
    ```
2. Get dependent model files ready.
    ```
    $scp -r xcdl190074:/group/dphi_edge/vai-1.3/dpu-for-rnn/models/* <Your_Path>/dsa/DPU-for-RNN/models/
    ```
    Then following [this guide](models/README.md) to get all dependent files ready.

3. Build rnn docker based on vitis-ai and launch the docker image.
    
    For CPU version, you can use prebuild version in github. For GPU, please firstly build Vitis AI GPU docker. 
    
    To build rnn docker:
    ```    
    $cd ../.. 
    $docker build --build-arg BASE_IMAGE=<Vitis AI Docker Image> -f dsa/DPU-for-RNN/Dockerfile.rnn -t <YOUR RNN Docker Image>  .
    e.g.:
    $cd ../..
    $docker build --build-arg BASE_IMAGE="xdock:5000/vitis-ai-gpu:1.3.343" -f dsa/DPU-for-RNN/Dockerfile.rnn -t xdock:5000/vitis-ai-gpu-rnn:1.3.343 . 
    ```    
    To launch the docker image:
    ```    
    $cd <Your_Path>/dsa/DPU-for-RNN/
    $bash docker_run.sh <YOUR RNN Docker Image>
    ```
4. Setup the Alveo U25 (or U50LV)
    ``` 
    $cd scripts
    $source setup_u25.sh # Alveo U25
    $source setup_u50lv.sh # Alveo U50LV
    ```
5. Execute commands as described in application README files. It indicates that your runs get correct output if it reports the same accurcy number as provided in readme file. 
    - [README: Customer Satisfaction](app/customer_satisfaction/README.md)
    - [README: Imdb Sentiment Detection](app/imdb_sentiment_detection/README.md)
    - [README: Open Information Extraction](app/open_information_extraction/README.md)
