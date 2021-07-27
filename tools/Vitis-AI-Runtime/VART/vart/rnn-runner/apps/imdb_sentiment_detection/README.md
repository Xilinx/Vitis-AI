# Run IMDB Sentiment Detection Network

```sh
cd /workspace/vart/rnn-runner/apps/imdb_sentiment_detection/

# [For U25]
cp /workspace/vart/rnn-runner/xclbin/u25/dpu.xclbin /usr/lib/dpu.xclbin

# [For U50LV]
cp /workspace/vart/rnn-runner/xclbin/u50lv/dpu.xclbin /usr/lib/dpu.xclbin
```

1. Download the model files
```sh
mkdir -p models/

cp /proj/rdi/staff/akorra/data/imdb_sentiment_detection/LSTM.h5 models/

# [For U25]
cp /proj/rdi/staff/akorra/data/imdb_sentiment_detection/u25/*.xmodel models/

# [For U50LV]
cp /proj/rdi/staff/akorra/data/imdb_sentiment_detection/u50lv/*.xmodel models/
```

1. Download the datasets
```sh
mkdir -p data
cp /proj/rdi/staff/akorra/data/imdb_sentiment_detection/IMDB.csv data/
cp /proj/rdi/staff/akorra/data/imdb_sentiment_detection/imdb.npz data/
cp /proj/rdi/staff/akorra/data/imdb_sentiment_detection/imdb_word_index.json data/
```

1. Setup the environment.
```sh
source ./setup.sh
```

1. Run the CPU mode with all transactions.
```sh
python run_cpu_e2e.py
```

1. Run the DPU mode with all transactions.
```sh
# For U25
TARGET_DEVICE=U25 python run_dpu_e2e.py

# For U50LV
TARGET_DEVICE=U50LV python run_dpu_e2e.py
```

1. run the dpu mode with multithread support
```sh
# For U25
TARGET_DEVICE=U25 python run_dpu_e2e_mt.py -n 4

# For U50LV
TARGET_DEVICE=U50LV python run_dpu_e2e_mt.py -n 4
````

> The accuracy for the end-to-end mdoel test should be: `0.8689`


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
