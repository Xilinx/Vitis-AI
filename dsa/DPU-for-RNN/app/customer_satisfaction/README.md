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
