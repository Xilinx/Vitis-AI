# vai_q_pytorch

### Introduction
vai_q_pytorch is short for Vitis AI Quantizer for Pytorch. It is a tool for neural network model optimization with Pytorch model input.
vai_q_pytorch is designed as a part of a standard platform for neural network deep compression. Base on this architecture, working stages of vai_q_pytorch follows:<br>
    1.  Parse neural network computation graph from Pytorch framework to Intermediate Representation (IR).<br>
    2.  Modify the computation graph based on IR. Pruning, quantization and graph optimization are handled in this stage.<br>
    3.  Deploy the modified IR on different platform, such as DPU and Xilinx AI Engine.<br>
    4.  Assemble the modified computation graph back to Pytorch. In that way abilities and tools in Pytorch such as pre-processing, post processing and distribution system can be used.<br>

### Supported

Python version 3.6 ~ 3.7.

Pytorch version 1.1 ~ 1.4.

GPU only. Currently Pytorch quantization tool only has GPU version.

Classification Models in Torchvision, Pytorch models in Xilinx Modelzoo. 

### Quick Start in Docker environment

If you work in Vitis-AI 1.2 docker, there is a conda environment "vitis-ai-pytorch", in which vai_q_pytorch package is already installed. 
In this conda environment, python version is 3.6, pytorch version is 1.1 and torchvision version is 0.3.0. You can directly start our "resnet18" example without installation steps.
If you want a different “python/pytorch/torchvision” version, install vai_q_pytorch from source code.
- Copy example/resnet18_quant.py to docker environment
- Download pre-trained [Resnet18 model](https://download.pytorch.org/models/resnet18-5c106cde.pth)
  ```shell
  wget https://download.pytorch.org/models/resnet18-5c106cde.pth -O resnet18.pth
  ```
- Prepare Imagenet validation images
- Modify default data_dir and model_dir in resnet18_quant.py
- [Optional] Evaluate float model
  ```shell
  python resnet18_quant.py --quant_mode 0
  ```
- Quantize, using a subset(200 images) of validation data for calibration. Because we are in quantize calibration process, the displayed loss and accuracy are meaningless.
  ```shell
  python resnet18_quant.py --quant_mode 1 --subset_len 200
  
  ```
- Evaluate quantized model
  ```shell
  python resnet18_quant.py --quant_mode 2
  ```

### Install from source code

Installation with Anaconda is suggested. 

To install vai_q_pytorch, do as follows:

##### Pre step 1 : set CUDA_HOME environment variable in .bashrc
If CUDA library is installed in /usr/local/cuda, add the following line into .bashrc. If CUDA is in other directory, change the line accordingly.

    export CUDA_HOME=/usr/local/cuda 

##### Pre step 2 : install Pytorch(1.1-1.4) and torchvision
Here take pytorch 1.1 and torchvision 0.3.0 as an example, detailed instructions for other versions are in [pytorch](https://pytorch.org/) website.

    pip install torch==1.1.0 torchvision==0.3.0 

##### Pre step 3 : install other dependencies
    pip install -r requirements.txt 

##### Now install the main component:
    cd ./pytorch_binding 
    python setup.py install (for user) 
    python setup.py develop (for developer) 

##### Verify the installation:
If the following command line does not report error, the installation is done.

    python -c "import pytorch_nndct"

To create deployed model, XIR library needs to be installed. If just run quantization and check the accuracy, this is not must. 
Refer to Vitis AI document for more information on deployment.

### vai_q_pytorch Tool Usage

vai_q_pytorch is designed to work as a Pytorch plugin. We provide simplest APIs to introduce our FPGA-friendly quantization feature.
For a well-defined model, user only need to add 2-3 lines to get a quantize model object.

##### Model pre-requirements for quantizer
- The model to be quantized should include forward method only. All other functions should be moved outside or move to a derived class. 
These functions usually work as pre-processing and post-processing. If they are not moved outside, 
our API will remove them in our quantized module, which will cause unexpected behaviour when forwarding quantized module. <br>
- The float model should pass "jit trace test". First set the float module to evaluation status, then use “torch.jit.trace” function to test the float model. Make sure the float module can pass the trace test. <br>

##### Add vai_q_pytorch APIs to float scripts
Before quantization, suppose there is a trained float model and some python scripts to evaluate model's accuracy/mAP. 
Quantizer API will replace float module with quantized module and normal evaluate function will encourage quantized module forwarding. 
Quantize calibration determines "quantize" op parameters in evaluation process if we set flag quant_mode to 1. 
After calibration, we can evaluate quantized model by setting quant_mode to 2.

Take resnet18_quant.py to demostrate how to add vai_q_pytorch APIs in float code. 
Xilinx [Pytorch Modelzoo](https://github.com/Xilinx/AI-Model-Zoo) includes float model and quantized model.
It is a good idea to check the difference between float and quantized script, like "code/test.py" and "quantize/quant.py" in ENet.

1. Import vai_q_pytorch modules <br>
   ```py
    from pytorch_nndct.apis import torch_quantizer, dump_xmodel
   ```
2. Generate a quantizer with quantization needed input and get converted model. <br>
   ```py
    input = torch.randn([batch_size, 3, 224, 224])
    quantizer = torch_quantizer(quant_mode, model, (input))
    quant_model = quantizer.quant_model
   ```
3. Forwarding with converted model. <br>

4. Output quantization result and deploy model. <br>
   ```py
    quantizer.export_quant_config()
    dump_xmodel()
   ```

##### Run and output results
Before running commands, let's introduce the log message in vai_q_pytorch. vai_q_pytorch log messages have special color and special keyword "NNDCT". 
"NNDCT" is our internal project name and we will change it later. vai_q_pytorch log message types include "error", "warning" and "note". 
Pay attention to vai_q_pytorch log messages to check the flow status.<br>
Run command with "--quant_mode 1" to quantize model.
```py
    python resnet18_quant.py --quant_mode 1 --subset_len 200
```
When doing calibration forward, we borrow float evaluation flow to minimize code change from float script. So there are loss and accuracy displayed in the end. 
They are meaningless, just skip them. Pay more attention to the colorful log messages with special keywords "NNDCT".

Another important thing is to contral iteration numbers during quantization and evaluation. 
Generally, 100-1000 images are enough for quantization and the whole validation set are required for evaluation. 
The iteration numbers can be controlled in the data loading part.
In this case, argument "subset_len" controls how many images used for network forwarding. 
But if the float evaluation script doesn't have an argument with similar role, it is better to add one, otherwise it should be changed manually.

If this quantization command runs successfully, two important files will be generated under output directory “./quantize_result”. 
```
    ResNet.py: converted vai_q_pytorch format model, 
    Quant_info.json: quantization steps of tensors got. (Keep it for evaluation of quantized model)
```
To evaluate quantized model, run the following command:
```shell
    python resnet18_quant.py --quant_mode 2
```
When this command finishes, the displayed accuracy is the right accuracy for quantized model. <br> 
Xmodel file for Vitis AI compiler will be generated under output directory “./quantize_result”. It will be further used to deploy this model to FPGA. 
```
    ResNet_int.xmodel: deployed model
```
If XIR is not installed, Xmodel file can't be generated, this command will raise error in the end. But the accuray can also be found in the output log.

### vai_q_pytorch APIs

The APIs are in module nndct/pytorch_binding/pytorch_nndct/apis/quant_api.py. Two main APIs are listed here.
##### Function torch_quantizer will create a quantizer.
```py
    def torch_quantizer(quant_mode,
                        module,
                        input_args,
                        state_dict_file,
                        output_dir,
                        bitwidth_w,
                        bitwidth_a)
```
    quant_mode: An integer that indicates which quantization mode the process is using. 0 for turning off quantization. 1 for calibration of quantization. 2 for evaluation of quantized model.
    Module: Float module to be quantized.
    Input_args: input tensor with the same shape as real input of float module to be quantized, but the values can be random number.
    State_dict_file: Float module pretrained parameters file. If float module has read parameters in, the parameter is not needed to be set.
    Output_dir: Directory for quantization result and intermediate files. Default is “quantize_result”.
    Bitwidth_w: Global weights and bias quantization bit width. Default is 8.
    Bitwidth_a: Global activation quantization bit width. Default is 8.
##### Function dump_xmodel will create deployed model. 
```py
    def dump_xmodel(output_dir, deploy_check)
```
    Output_dir: Directory for quantizapyttion result and intermediate files. Default is “quantize_result”
    Depoly_check: Flags to control dump of data for accuracy check. Default is False.




