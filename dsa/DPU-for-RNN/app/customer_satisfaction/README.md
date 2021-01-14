##### Run Customer Satisfaction Network

1. Compile the dpu4rnn library
    ```    
    $./build_libdpu4rnn.sh
    ```
2. Setup the machine
    ```
    $source ./setup.sh
    ```
3. run the cpu mode with all trasactions
    ```
    $python run_cpu_e2e.py
    ```
4. run the dpu mode with all transactions
    ```
    $python run_dpu_e2e.py
    ```
5. The accuracy for the end-to-end model test should be:
   0.9565.

#### License
Copyright 2019 Xilinx Inc.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.


