# Run IMDB sentiment detection model based on GRU.

1. **This application should be run inside Vitis-AI-RNN docker container.**
    ```sh
    cd ${VAI_HOME}/demo/DPU-for-RNN/rnn_u25_u50lv/apps/gru_sentiment_detection/
    ```

1. Setup the environment
    ```sh
    # For U50LV
    TARGET_DEVICE=U50LV source ./setup.sh
    ```

1. Run the cpu mode with all trasactions
    ```sh
    python run_e2e.py -d CPU
    ```

1. Run the dpu mode with all transactions
    ```sh
    # For U50LV, run with single runner
    python run_e2e.py -d U50LV -n 1

    # For U50LV, run with two runners
    python run_e2e.py -d U50LV -n 2
    ```

> The accuracy for the end-to-end model test on DPU should be around `81%`

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
