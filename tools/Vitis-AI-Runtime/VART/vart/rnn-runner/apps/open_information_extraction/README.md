# Run Open Information Extraction Network

1. Copy compiled model files
    ``` bash
    cd /workspace/vart/rnn-runner/apps/open_information_extraction
    mkdir -p models

    # For U50LV
    cp -r /proj/rdi/staff/akorra/data/openie/* models/

    # For U25
    cp -r ../u25/open_information_extraction/* models/
    cp utils/u25_config.json model/config.json
    ```

1. Download model files
    ``` bash
    wget https://allennlp.s3.amazonaws.com/models/openie-model.2018-08-20.tar.gz
    mkdir -p weights
    tar -xzvf openie-model.2018-08-20.tar.gz -C weights
    ```

1. Download the allennlp.
    ``` bash
    git clone https://github.com/allenai/allennlp out_project
    cd out_project
    git checkout f3083c8fb9150f07e3ca98bb3ea9368a081df028
    cd ..
    mv out_project/allennlp .
    git clone https://github.com/gabrielStanovsky/supervised_oie_wrapper
    cp supervised_oie_wrapper/src/format_oie.py .
    cp supervised_oie_wrapper/src/run_oie.py .
    ```
2. Copy files for benchmarking.
    ```
    git clone https://github.com/gabrielStanovsky/oie-benchmark
    cp utils/moveConf.py  oie-benchmark/
    cp utils/benchmark.py oie-benchmark/
    cp utils/tabReader.py oie-benchmark/oie_readers/
    cp utils/test.oie     oie-benchmark/oie_corpus/
    ```
4. Setup the machine.
    ```
    source setup.sh
    ```

3. Download NLTK packages for testing
    ``` bash
    python -m nltk.downloader -d ~/nltk_data wordnet stopwords
    ```

    > :warning: **WARNING** : If your deployment machine doesn't have internet, please download the packages on a different machine and copy the files to deployment machine as given below.

    ``` bash
    # 1. install nltk in your python environment. Use conda/pip
    conda install nltk

    # 2. Download wordnet corpora to ~/nltk_data
    python -m nltk.downloader -d ~/nltk_data wordnet stopwords

    # 3. Copy ~/nltk_data to home_directory in your deployment machine
    scp -r ~/nltk_data <deployment_machine>:~/

    # 4. On deployment machine, comment out oie-benchmark/matcher.py:L7 to avoid downloading again.
    # nltk.download('wordnet', quiet=True)
    ```

5. Create the test text.
    ``` bash
    mkdir -p test
    python convert.py
    ```
6. Run the CPU mode.
    ```
    ./run_cpu_one_trans.sh # one transaction
    ./run_cpu.sh           # all transactions
    ```
7. Run the DPU mode.
    ``` bash
    # For U50LV
    TARGET_DEVICE=U50LV ./run_dpu.sh

    # For U25
    TARGET_DEVICE=U25 ./run_dpu.sh
    ```
8. The accuracy for the end-to-end open information extraction model test should be:
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


