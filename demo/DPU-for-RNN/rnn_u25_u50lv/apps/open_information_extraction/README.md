# Run Open Information Extraction Network

1. **This application should be run inside Vitis-AI-RNN docker container.**
    ```sh
    cd ${VAI_HOME}/demo/DPU-for-RNN/rnn_u25_u50lv/apps/open_information_extraction/
    ```

1. Setup the environment
    ```sh
    # For U25
    TARGET_DEVICE=U25 source ./setup.sh

    # For U50LV
    TARGET_DEVICE=U50LV source ./setup.sh
    ```

1. Run the CPU mode.
    ```sh
    ./run_cpu_one_trans.sh # one transaction
    ./run_cpu.sh           # all transactions
    ```
1. Run the DPU mode.
    ``` sh
    # For U50LV
    TARGET_DEVICE=U50LV ./run_dpu.sh

    # For U25
    TARGET_DEVICE=U25 ./run_dpu.sh
    ```
1. The accuracy for the end-to-end open information extraction model test should be around:
    - auc : 0.5875
    - f1 : 0.7720

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


