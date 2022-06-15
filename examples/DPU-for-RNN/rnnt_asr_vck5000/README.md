# RNN-T Demo on Versal
## Introduction
This demo shows one RNN-T model based ASR solution on Xilinx Versal device VCK5000. Versal is the first Adaptive Compute Acceleration Platform (ACAP). It is a fully software-programmable heterogeneous compute platform that combines Scalar Engines, Adaptable Engines, and Intelligent Engines to achieve dramatic performance improvements over FPGA and CPU implementations among different applications. Please find more information on the [Xilinx Versal website](https://www.xilinx.com/products/silicon-devices/acap/versal.html). RNN-T is a sequence-to-sequence model that continuously processes input samples and streams output symbols. The speech recognition model used here is a modified RNN-T model and belongs to the MLPerf Inference benchmark suite. More details about the model can be found from the [mlcommons inference repo](https://github.com/mlcommons/inference/tree/r0.7/speech_recognition/rnnt).

The hardware kernel is a 40 AIE cores design. INT8 Matrix-Matrix multiplications are performed on AIE cores and other functions are implemented in Programmable Logic (PL) with INT16 precision. The following table shows the total resource utilization for the kernel and platform. URAM resources are mainly used as weights buffer. It will be shared if multiple kernels are instantiated.

|           | CLB LUTs | Registers | Block RAM | URAM | DSP Slices | AIE Cores |
|:---------:|:--------:|:---------:|:---------:|:----:|:----------:|:---------:|
| Available | 899712   | 1799424   |  967      |463   |1968        |400        |
| Utilized  |169163(18.8%)|241657(13.43%)|197(20.37%)|332(71.71%)|82(4.17%)|40(10%)|

The Quantizer and Compiler are not ready now. The process of mix-precision quantization and instruction generation are completed manually for this demo.

Please take the following steps to run the RNN-T based ASR demo on Versal VCK5000.

## RNN-T demo dependency setup

```shell
sh scipts/copyfile.sh
```
Download the models
```shell
wget https://www.xilinx.com/bin/public/openDownload?filename=best_lstm1616else816.pt -O my_work_dir/best_lstm1616else816.pt
wget https://www.xilinx.com/bin/public/openDownload?filename=rnnt.bin -O model/rnnt.bin
```
We use the [Librispeech  dev clean dataset](http://www.openslr.org/resources/12/dev-clean.tar.gz) as the test dataset. Please download it to the current folder.
```shell
wget http://www.openslr.org/resources/12/dev-clean.tar.gz 
```

Download the xclbin file
```shell
wget https://www.xilinx.com/bin/public/openDownload?filename=dpu4rnn_vck5000_xclbin.tar.gz
```
Then decompress the file and copy file 'xvrnn.hw.xclbin' to the xclbin directory: DPU-for-RNN/rnnt_asr_vck5000/xclbin/.

## Start Docker and activate the conda environment

```shell
sh scripts/docker_run.sh
conda activate rnn-pytorch-1.7.1
```

## Test dataset preparation

The scripts will generate a summary json file of the dataset.
Besides, the number of workers in dataloader is 0 by default, you may change it in dataset.py to increase loading data speed.

```shell 
sh scripts/dataset_prepare.sh
```

## Compile lib

You don't need to compile it in the docker environment because the library is already prepared. 
If you want to recompile the library, you should do it in the docker by the following scripts.

```shell
cd hwlib/libxvrnn/
sh make.sh
```
If you want to compile your own library, you need to set your related python paths in hwlib/libxvrnn/CMakeLists.txt


## Run librispeech dev clean  on CPU 

```shell
sh scipts/run_dataset_cpu.sh
```

## Run librispeech dev clean on DPU 

```shell
sh scripts/run_dataset_dpu.sh
```


## Run long wav test

```shell
sh scripts/run_audio_dpu.sh
```

## License
Copyright 2019 Xilinx Inc.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at
```
http://www.apache.org/licenses/LICENSE-2.0
```
Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.




