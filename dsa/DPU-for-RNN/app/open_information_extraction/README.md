##### Run OPen Information Extraction Network

1. Download the allennlp.
    ```    
    $cd app/open_information_extraction
    $git clone https://github.com/allenai/allennlp out_project
    $cd out_project
    $git checkout f3083c8fb9150f07e3ca98bb3ea9368a081df028
    $cd ..
    $mv out_project/allennlp .
    $git clone https://github.com/gabrielStanovsky/supervised_oie_wrapper
    $mv supervised_oie_wrapper/src/format_oie.py .
    $mv supervised_oie_wrapper/src/run_oie.py .
    ```
2. Copy files for benchmarking.
    ```
    $git clone https://github.com/gabrielStanovsky/oie-benchmark
    $cp backup/moveConf.py  oie-benchmark/
    $cp backup/benchmark.py oie-benchmark/
    $cp backup/tabReader.py oie-benchmark/oie_readers/
    $cp backup/test.oie     oie-benchmark/oie_corpus/
    ```
3. Compile the dpu4rnn library.
    ```
    $./build_libdpu4rnn.sh
    ```
4. Setup the machine.
    ```
    $source setup.sh
    ```
5. Run the CPU mode.
    ```
    $./run_cpu_one_trans.sh # one transaction
    $./run_cpu.sh           # all transactions
    ```
6. Run the DPU mode.
    ```
    $./run_dpu_one_trans.sh  # one transaction
    $./run_dpu.sh            # all transactions
    ```
7. The accuracy for the end-to-end open information extraction model test should be:
    - auc : 0.5875
    - f1 : 0.7720

