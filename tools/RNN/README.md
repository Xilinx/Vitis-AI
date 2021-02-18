## Vitis AI RNN Tool

Vitis AI RNN tool is designed to support different RNN networks, including RNN, LSTM, Bi-directional LSTM and GRU. 
Currently only standard LSTM is supported.

Vitis AI RNN tool includes rnn quantizer and rnn compiler. 
The quantizer performs fixed-point int16 quantization for model parameters and input. 
The compiler generates instructions based on the target network structure and RNN hardware architecture. 
So far, this tool chain only supports standard LSTM, and the generated instructions can only be deployed in U25 card and U50 card. 
For U25 card, the weights of model can’t exceed 4MB, and for U50 card, the weights can’t exceed 8MB. 
The workflow of the tool chain shows as follows:

1. Import RNN model and input argument to RNN quantizer, get the quantization information of the model.

2. Using the quantization information, RNN quantizer converts float model to quantized model. Get the Xmodel file, which is the representation of the model’s computation process.

3. RNN compiler compiles the Xmodel file to hardware instructions, which will be deployed on FPGA.

## Quick Start

Take the sentiment detection model as an example, we describe a quick way to use vitis AI RNN tool.

### Launch RNN quantizer and compiler docker container
Launch the RNN quantizer and compiler docker container and activate conda environment "vitis-ai-lstm". All the requirements for RNN quantizer and compiler are ready there. Note that RNN tools now only have GPU version, so please use GPU docker. Refer to [Vitis AI homepage](https://github.com/Xilinx/Vitis-AI) for building GPU dockers.
```
./docker_run.sh xilinx/vitis-ai-gpu:latest
conda activate vitis-ai-lstm
```

### Quantization
1. In docker environment, copy model and scripts to the quantizer working directory. The example is available in rnn_quantizer/example/lstm_quant_tensorflow.
```
mkdir -p /workspace/rnn_workflow/sentiment_detection/quantizer
cp /workspace/tools/RNN/rnn_quantizer/example/lstm_quant_tensorflow/* /workspace/rnn_workflow/sentiment_detection/quantizer/
```
The contents in the working directory are showed as follows.
```
pretrained.h5: pretrained model for sentiment detection.
quantize_lstm.py: python script to run quantization of the model
run_quantizer.sh: shell script to run python script
```
2. In docker environment, run quantization and get the results.
```
cd /workspace/rnn_workflow/sentiment_detection/quantizer
sh run_quantizer.sh
```
If this quantization command runs successfully, two important files and one import subdirectory are generated in the output directory "./quantize_result".
```
rnn_cell_0.py: converted format model
quant_info.json: quantization steps of tensors
xmodel: subdirectory that contain deployed model
```

### Compilation 
1. In docker environment, copy the compiler script to the compiler directory. The script is available in rnn_compiler/examples/lstm_compiler_test.py.
```
mkdir -p /workspace/rnn_workflow/sentiment_detection/compiler
cp /workspace/tools/RNN/rnn_compiler/examples/lstm_compiler_test.py /workspace/rnn_workflow/sentiment_detection/compiler/
```
2. In docker environment, run the compilation and get the result.

For U25 card, run the following commands.
```
cd /workspace/rnn_workflow/sentiment_detection/compiler
python lstm_compiler_test.py --model_path ../quantizer/quantize_result/xmodel --device u25
```
This command will generate the result as follows. Two subdirectories are generated, computation graphs are stored in the subdirectory “CompileData", and instructions to deployed in hardware are stored in subdirectory “Instructions”.
```
|-- CompileData
|   |-- model_config.json
|   `-- rnn_cell_0
|       |-- cycle_ops.json
|       |-- invalid_ops.json
|       |-- nodes_edge.json
|       `-- nodes_info.json
|-- Instructions
|   |-- ddr_init_orign.txt
|   |-- ddr_init_str.txt
|   |-- fp.json
|   |-- instr_ac.txt
|   `-- u25_ddr_bin
```

For U50 card, run the following commands.
```
cd /workspace/rnn_workflow/sentiment_detection/compiler
python lstm_compiler_test.py --model_path ../quantizer/quantize_result/xmodel --device u50
```
And get the result as follows.
```
-- CompileData
|   |-- model_config.json
|   `-- rnn_cell_0
|       |-- cycle_ops.json
|       |-- invalid_ops.json
|       |-- nodes_edge.json
|       `-- nodes_info.json
|-- Instructions
|   |-- config.json
|   |-- ddr_bin_3k
|   |-- ddr_bin_4k
|   |-- ddr_init_orign.txt
|   |-- ddr_init_str.txt
|   |-- fp.json
|   |-- instr_ac_3.txt
|   |-- instr_ac_4.txt
|   |-- is_3
|   `-- is_4
```

### Copy Data to RNN runtime docker
1. To run the generated instructions in hardware, another runtime docker should be built. The docker can be built and launched as follows.
```
docker build --build-arg BASE_IMAGE="xilinx/vitis-ai-cpu:latest" -f dsa/DPU-for-RNN/Dockerfile.rnn -t xilinx/vitis-ai-cpu-rnn:latest . 
cd dsa/DPU-for-RNN/
bash docker_run.sh xilinx/vitis-ai-cpu-rnn:latest
```
2. Download the required data. Besides the data below, the imdb.csv should also be downloaded. You can register an account on this website (https://www.kaggle.com/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews) and download it.
```
wget https://storage.googleapis.com/tensorflow/tf-keras-datasets/imdb.npz
wget https://storage.googleapis.com/tensorflow/tf-keras-datasets/imdb_word_index.json
```
3. Copy the required data, pretrained model and instructions to the hardware-deployed directory.

For U25 card, use the following commands.
```
mkdir -p dsa/DPU-for-RNN/models/u25/customer_satisfaction/
cp ./IMDB.csv dsa/DPU-for-RNN/models/u25/imdb_sentiment_detection/IMDB.csv
cp ./imdb.npz dsa/DPU-for-RNN/models/u25/imdb_sentiment_detection/imdb.npz
cp ./imdb_word_index.json dsa/DPU-for-RNN/models/u25/imdb_sentiment_detection/imdb_word_index.json
cp rnn_workflow/sentiment_detection/quantizer/pretrained.h5 dsa/DPU-for-RNN/models/u25/imdb_sentiment_detection/LSTM.h5
cp rnn_workflow/sentiment_detection/compiler/Instructions/u25_ddr_bin dsa/DPU-for-RNN/models/u25/imdb_sentiment_detection/sentiment_0.bin
cp rnn_workflow/sentiment_detection/compiler/Instructions/fp.json dsa/DPU-for-RNN/models/u25/imdb_sentiment_detection/sentiment.json
```
For U50 card, use the following commands.
```
mkdir -p dsa/DPU-for-RNN/models/u50lv/customer_satisfaction/
cp ./IMDB.csv dsa/DPU-for-RNN/models/u50lv/imdb_sentiment_detection/IMDB.csv
cp ./imdb.npz dsa/DPU-for-RNN/models/u50lv/imdb_sentiment_detection/imdb.npz
cp ./imdb_word_index.json dsa/DPU-for-RNN/models/u50lv/imdb_sentiment_detection/imdb_word_index.json
cp rnn_workflow/sentiment_detection/quantizer/pretrained.h5 dsa/DPU-for-RNN/models/u50lv/imdb_sentiment_detection/LSTM.h5
cp rnn_workflow/sentiment_detection/compiler/Instructions/config.json dsa/DPU-for-RNN/models/u50lv/imdb_sentiment_detection/config.json
cp rnn_workflow/sentiment_detection/compiler/Instructions/is_3 dsa/DPU-for-RNN/models/u50lv/imdb_sentiment_detection/is_3
cp rnn_workflow/sentiment_detection/compiler/Instructions/is_4 dsa/DPU-for-RNN/models/u50lv/imdb_sentiment_detection/is_4
cp rnn_workflow/sentiment_detection/compiler/Instructions/ddr_bin_3k dsa/DPU-for-RNN/models/u50lv/imdb_sentiment_detection/sentiment_0.bin
cp rnn_workflow/sentiment_detection/compiler/Instructions/ddr_bin_4k dsa/DPU-for-RNN/models/u50lv/imdb_sentiment_detection/sentiment_1.bin
cp rnn_workflow/sentiment_detection/compiler/Instructions/fp.json dsa/DPU-for-RNN/models/u50lv/imdb_sentiment_detection/sentiment.json
```
4. Follow [DPU-for-RNN instructions](../../dsa/DPU-for-RNN) to run model in FPGA 
