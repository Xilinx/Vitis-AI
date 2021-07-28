
# Setup
```sh
cd vart
./docker_run.sh xilinx/vitis-ai-cpu:latest

cd /workspace && ./cmake.sh --cmake-options="-DENABLE_RNN_RUNNER=ON --DENABLE_CPU_RUNNER=OFF -DENABLE_SIM_RUNNER=OFF" && cd -
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/.local/Ubuntu.18.04.x86_64.Debug/lib

# U25
sudo cp /workspace/rnn-runner/xclbin/u25/dpu.xclbin /usr/lib/dpu.xclbin

# U50
sudo cp /workspace/rnn-runner/xclbin/u50lv/dpu.xclbin /usr/lib/dpu.xclbin
```

## Download model files

### Download U25 model
```sh
mkdir -p /workspace/rnn-runner/test/models
wget -O u25-2020-12-16.tar.gz https://www.xilinx.com/bin/public/openDownload?filename=u25-2020-12-16.tar.gz
tar -xvf u25-2020-12-16.tar.gz -C /workspace/rnn-runner/test/models && rm -rf u25-2020-12-16.tar.gz

cp -r <xcd>:/group/dphi_edge/vai-1.4-rnn-xmodel/ /workspace/rnn-runner/test
```

### Download U50 model
```sh
mkdir -p /workspace/rnn-runner/test/models
wget -O u50lv-2020-12-16.tar.gz https://www.xilinx.com/bin/public/openDownload?filename=u50lv-2020-12-16.tar.gz
tar -xvf u50lv-2020-12-16.tar.gz -C /workspace/rnn-runner/test/models && rm u50lv-2020-12-16.tar.gz

cp -r <xcd>:/group/dphi_edge/vai-1.4-rnn-xmodel/ /workspace/rnn-runner/test
```


## Get configuration for U25

### Customer Satisfaction
```sh
cp /workspace/rnn-runner/apps/customer_satisfaction/utils/u25_config.json /workspace/rnn-runner/test/models/u25/customer_satisfaction/config.json
```

### IMDB Sentiment Detection
```sh
cp /workspace/rnn-runner/apps/imdb_sentiment_detection/utils/u25_config.json /workspace/rnn-runner/test/models/u25/imdb_sentiment_detection/config.json
```

## OpenIE
```sh
cp /workspace/rnn-runner/apps/open_information_extraction/utils/u25_config.json /workspace/rnn-runner/test/models/u25/open_information_extraction/config.json
```

## Copy test data

```sh
cd /workspace/rnn-runner/test
cp -r <xhd>:/proj/sdxapps/users/akorra/datasets/rnn-runner-test-data.tar.gz .
cp -r <xsj>:/proj/xsjhdstaff4/akorra/datasets/rnn-runner-test-data.tar.gz .
cp -r <xbjjmphost>:/home/akorra/rnn-runner-test-data.tar.gz .
cp -r <xcd>:/proj/rdi/staff/akorra/datasets/rnn-runner-test-data.tar.gz .
tar -xzvf rnn-runner-test-data.tar.gz -C /workspace/rnn-runner/test && rm rnn-runner-test-data.tar.gz
```

# Run tests

## U25 tests

> ./test_rnn_runner_u25 <device_name> <model_dir> <model_name> <test_dir> <num_sequences> <num_threads> <num_batches>

### Customer Satisfaction

```sh
cd ~/build/build.Ubuntu.18.04.x86_64.Debug/workspace/rnn-runner/
export MODEL_DIR="/workspace/rnn-runner/test/models/u25/customer_satisfaction/"
export TEST_DIR="/workspace/rnn-runner/test/data/u25/satis"
./test_rnn_runner_u25 U25 $MODEL_DIR satisfaction $TEST_DIR 25 1 1
```

### IMDB Sentiment Detection

```sh
cd ~/build/build.Ubuntu.18.04.x86_64.Debug/workspace/rnn-runner/
export MODEL_DIR="/workspace/rnn-runner/test/models/u25/imdb_sentiment_detection/"
export TEST_DIR="/workspace/rnn-runner/test/data/u25/sent"
./test_rnn_runner_u25 U25 $MODEL_DIR sentiment $TEST_DIR 500 1 1
```

### OpenIE

```sh
cd ~/build/build.Ubuntu.18.04.x86_64.Debug/workspace/rnn-runner/
export MODEL_DIR="/workspace/rnn-runner/test/models/u25/open_information_extraction/"

# num_sequences can be any of [36, 48, 59, 100, 141, 198]
export TEST_DIR="/workspace/rnn-runner/test/data/u25/oie/100"
./test_rnn_runner_u25 U25 $MODEL_DIR openie $TEST_DIR 100 1 1
```


## U50 tests

> ./test_rnn_runner_u50 <device_name> <model_dir> <model_name> <test_dir> <num_sequences> <num_threads> <num_batches>

### Customer Satisfaction

```sh
cd ~/build/build.Ubuntu.18.04.x86_64.Debug/workspace/rnn-runner/
export TEST_DIR="/workspace/rnn-runner/test/data/u50lv/satis"

export MODEL_DIR="/workspace/rnn-runner/test/xmodels/u50/lstm_customer_satisfaction/without_hbm_opt/"
# ./test_rnn_runner <xmodel_dir> <test_dir> <num_sequences> <num_threads> <num_batches>
./test_rnn_runner_u50 $MODEL_DIR $TEST_DIR 25 1 1
```

### IMDB Sentiment Detection

```sh
cd ~/build/build.Ubuntu.18.04.x86_64.Debug/workspace/rnn-runner/
export TEST_DIR="/workspace/rnn-runner/test/data/u50lv/sent"

export MODEL_DIR="/workspace/rnn-runner/test/xmodels/u50/lstm_sentiment_detection/without_hbm_opt/"
# ./test_rnn_runner <xmodel_dir> <test_dir> <num_sequences> <num_threads> <num_batches>
./test_rnn_runner_u50 $MODEL_DIR $TEST_DIR 500 1 1
```

### OpenIE

```sh
cd ~/build/build.Ubuntu.18.04.x86_64.Debug/workspace/rnn-runner/

# num_sequences can be any of [36, 59]
export TEST_DIR="/workspace/rnn-runner/test/data/u50lv/oie/36"

export MODEL_DIR="/workspace/rnn-runner/test/xmodels/u50/openie-new/without_hbm_opt/"
# ./test_rnn_runner <xmodel_dir> <test_dir> <num_sequences> <num_threads> <num_batches>
./test_rnn_runner_u50 $MODEL_DIR $TEST_DIR 36 1 1
```
