##### Run IMDB Sentiment Detection Network

1. Compile the dpu4rnn library.
    ```    
    $./build_libdpu4rnn.sh
    ```
2. Setup the machine.
    ```
    $source ./setup.sh
    ```
3. Run the CPU mode with all transactions.
    ```
    $python run_cpu_e2e.py
    ```
4. Run the DPU mode with all transactions.
    ```
    $python run_dpu_e2e.py 
    ```
5. The accuracy for the end-to-end mdoel test should be:
    0.8689.
