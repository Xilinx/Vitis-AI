
﻿<table class="sphinxhide">
 <tr>
   <td align="center"><img src="https://raw.githubusercontent.com/Xilinx/Image-Collateral/main/xilinx-logo.png" width="30%"/><h1>Vitis AI</h1><h0>Adaptable & Real-Time AI Inference Acceleration</h0>
   </td>
 </tr>
</table>

# RNN Quantizer

### Introduction
RNN quantizer is designed to quantize recurrent neural network(RNN) models. Currently RNN quantizer only supports standard LSTM.

### Supported and Limitations

* Python
1. Support version 3.6 ~ 3.7. 
    
* Pytorch
1. Support version 1.1 ~ 1.9.1. 
2. Data Parallelism is NOT supported.

* TensorFlow
1. Support version 2.0 ~ 2.6. 
    
* Models
1. Standard LSTM Models. 
    
### Quick Start in Docker environment

If you work in Vitis-AI 2.0 docker, there is two conda environments, "vitis-ai-rnn-pytorch" and "vitis-ai-rnn-tensorflow", in which RNN quantizers for pytorch and tensorflow are already installed. 
In this conda environment "vitis-ai-rnn-pytorch", python version is 3.6, pytorch version is 1.7.1. And in the conda environment "vitis-ai-rnn-tensorflow", python version is 3.6, tensorflow version is 2.0. You can directly start lstm example without installation steps.
- Copy example/lstm_quant_pytorch to docker environment
- Quantize, using a subset(1000 sequences) of validation data for calibration. Because we are in quantize calibration process, the displayed loss and accuracy are meaningless.
  ```shell
  cd example/lstm_quant_pytorch
  python quantize_lstm.py --quant_mode calib --subset_len 1000
  ```
- Evaluate quantized model and export xmodel
  ```shell
  python quantize_lstm.py --quant_mode test --subset_len 1000
  ```
  
### Install from source code

Installation with Anaconda is suggested. And if there is an old version of RNN quantizer in the conda enviorment, suggest you remove all of its related files before install the new version. 

To install pytorch tools of RNN quantizer, do as follows:

##### Pre step 1 : CUDA_HOME environment variable in .bashrc
For GPU version, if CUDA library is installed in /usr/local/cuda, add the following line into .bashrc. If CUDA is in other directory, change the line accordingly.

    export CUDA_HOME=/usr/local/cuda 

##### Pre step 2 : install Pytorch(1.1-1.9.1) and torchvision
Here take pytorch 1.7.1 and torchvision 0.8.2 as an example, detailed instructions for other versions are in [pytorch](https://pytorch.org/) website.

    pip install torch==1.7.1 torchvision==0.8.2 

##### Pre step 3 : install other dependencies
    pip install -r requirements.txt 

##### Now install the main component:
    cd ./pytorch_binding 
    python setup.py install (for user) 
    python setup.py develop (for developer) 

##### Verify the installation:
If the following command line does not report error, the installation is done.

    python -c "import pytorch_nndct"

To install tensorflow tools of RNN quantizer, do as follows:

##### Pre step 1 : install tensorflow 2.0 gpu version

    pip install tensorflow_gpu==2.0.0 

##### Pre step 2 : compile GPU acceleration kernel
    cd ..
    mkdir build
    cd build
    cmake ..
    make -j10

##### Now install the main component:
    cd ../tensorflow
    pip install -r requirements.txt 
    python setup.py install 

##### Verify the installation:
If the following command line does not report error, the installation is done.

    python -c "import tf_nndct"

To create deployed model, XIR library needs to be installed. If just run quantization and check the accuracy, this is not must. 
Refer to Vitis AI document for more information on deployment.

**Note:**<br>
If pytorch version you installed < 1.4, import pytorch_nndct before torch in your script. This is caused by a pytorch bug before version 1.4.
Refer to Pytorch github issue [28536](https://github.com/pytorch/pytorch/pull/28536) and [19668](https://github.com/pytorch/pytorch/issues/19668) for details. 
```python
import pytorch_nndct
import torch
```

### RNN quantizer Tool Usage

We provide simplest APIs to introduce our FPAG-friendly quantization feature. For a well-defined model, user only need to add 2-3 lines to get a quantize model object.

##### Add RNN quantizer APIs to float scripts
Before quantization, suppose there is a trained float model and some python scripts to evaluate model's accuracy/mAP. 
Quantizer API will replace float module with quantized module and normal evaluate function will encourage quantized module forwarding. 
Quantize calibration determines "quantize" op parameters in evaluation process if we set flag quant_mode to "calib". 
After calibration, we can evaluate quantized model by setting quant_mode to "test".

##### Pytorch
An example is available in example/lstm_quant_pytorch/quantize_lstm.py.

1. Import the PyTorch quantizer modules <br>
   ```py
    from pytorch_nndct.apis import torch_quantizer
   ```
2. Generate a quantizer with quantization and get the converted model. <br>
   ```py
    quantizer = torch_quantizer(quant_mode=args.quant_mode, 
                            module=model, 
                            bitwidth=16, 
                            lstm=True)
    model = quantizer.quant_model
   ```
3. Forward a neural network with the converted model. <br>
   ```py
    acc = test(model, DEVICE, test_loader)
   ```
4. Output quantization result and deploy model. <br>
   ```py
   if args.quant_mode == 'calib':
      quantizer.export_quant_config()
    if args.quant_mode == 'test':
      quantizer.export_xmodel(deploy_check=True)
   ```

##### TensorFlow
An example is available in example/lstm_quant_tensorflow/quantize_lstm.py.

1. Import the TensorFlow quantizer modules <br>
   ```py
    from tf_nndct.quantization.api import tf_quantizer
   ```
2. Generate a quantizer with quantization needed input, and the batch size of input data must be 1,  then get the converted model. <br>
   ```py
    single_batch_data = X_test[:1, ]
    input_signature = tf.TensorSpec(single_batch_data.shape, tf.int32)
    quantizer = tf_quantizer(model, 
                            input_signature, 
                            quant_mode=args.quant_mode,
                            bitwidth=16)
    rebuilt_model = quantizer.quant_model
   ```
3. Forward a neural network with the converted model. <br>
   ```py
    output = rebuilt_model(X_test[start: end])
   ```
4. Output the quantization result and deploy the model. When dumping the outputs, the batch size of the input data must be 1.  <br>
   ```py
    if args.quant_mode == 'calib':
      quantizer.export_quant_config()
    elif args.quant_mode == 'test':
      quantizer.dump_xmodel()
      quantizer.dump_rnn_outputs_by_timestep(X_test[:1])
   ```

##### Run and output results
Take the PyTorch version as an example.<br>
* Run command with "--quant_mode calib" to quantize model.
```py
    python quant_lstm.py --quant_mode calib --subset_len 1000
```
When calibrating forward, borrow the float evaluation flow to minimize code change from float script. If there are loss and accuracy messages displayed in the end, you can ignore them. Note the colorful log messages with the special keyword, "VAIQ_*".

If this quantization command runs successfully, two important files are generated in the output directory “./quantize_result”.
```
    Lstm_StandardLstmCell_layer_0_forward.py: converted format model,
    quant_info.json: quantization steps of tensors got. (Keep it for evaluation of quantized model)
```
* To evaluate quantized model, run the following command:
```shell
    python quant_lstm.py --quant_mode test --subset_len 1000
```
The accuracy displayed after the command has executed successfully is the right accuracy for the quantized model. The Xmodel file for the compiler is generated in the output directory, ./quantize_result/xmodel. <br> 
```
    Lstm_StandardLstmCell_layer_0_forward_int.xmodel: deployed model
```
In conda env vitis-ai-lstm in Vitis-AI docker, XIR is ready. But if RNN quantizer is installed by source code, it needs to install XIR in advance.<br>
If XIR is not installed, xmodel file can't be generated, this command will raise error in the end.

## RNN quantizer main APIs

#### Pytorch
The APIs are in module [pytorch_binding/pytorch_nndct/apis.py](pytorch_binding/pytorch_nndct/apis.py):
##### Function torch_quantizer will create a quantizer.
```py
class torch_quantizer(): 
    def __init__(self,
                 quant_mode: str, # ['calib', 'test']
                 module: torch.nn.Module,
                 input_args: Union[torch.Tensor, Sequence[Any]] = None,
                 state_dict_file: Optional[str] = None,
                 output_dir: str = "quantize_result",
                 bitwidth: int = 8,
                 mix_bit: bool = False,
                 device: torch.device = torch.device("cuda"),
                 lstm: bool = False,
                 app_deploy: str = "CV",
                 qat_proc: bool = False,
                 custom_quant_ops: List[str] = None):
```
    quant_mode: A string that indicates which quantization mode the process is using. "calib" for calibration of quantization. "test" for evaluation of quantized model.
    module: Float module to be quantized.
    state_dict_file: Float module pretrained parameters file. If float module has read parameters in, the parameter is not needed to be set.
    output_dir: Directory for quantization result and intermediate files. Default is “quantize_result”.
    bitwidth: Global quantization bit width. Default is 8.
    device: Run model on GPU or CPU.
    lstm: Flag to control whether this is an LSTM model. Default is false.
    qat_proc: Turn on quantization-aware-training (QAT).

##### Get the quantized model
```py
  @property
  def quant_model(self)
```
##### Export quantization steps information for tensors to be quantized
```py
  def export_quant_config(self)
```
##### Export quantization steps information for tensors to be quantized
```py
  def export_xmodel(self, output_dir="quantize_result", deploy_check=False)
```
    output_dir: Directory to save the xmodel files. Default is “quantize_result”.
    deploy_check: Flag to control whether to deploy simulation data.

#### TensorFlow
The APIs are in module [tensorflow/tf_nndct/quantization/api.py](tensorflow/tf_nndct/quantization/api.py) and [tensorflow/tf_nndct/quantization/quantizer.py](tensorflow/tf_nndct/quantization/quantizer.py):
##### Function tf_quantizer does quantization process of LSTM model..
```py
def tf_quantizer(model,
                 input_signature,
                 quant_mode: str = "calib",
                 output_dir: str = "quantize_result",
                 bitwidth: int = 8)
```
    model: Float module to be quantized.
    input_signature: Input tensor with the same shape as real input of float module to be quantized, but the values can be random number.
    quant_mode: A string that indicates which quantization mode the process is using. "calib" for calibration of quantization. "test" for evaluation of quantized model.
    output_dir: Directory for quantization result and intermediate files. Default is “quantize_result”.
    bitwidth: Global quantization bit width. Default is 8.

##### Get the quantized model
```py
  @property
  def quant_model(self)
```
##### Export quantization steps information for tensors to be quantized
```py
  def export_quant_config(self)
```
##### Export Xmodel files for compilation
```py
  def dump_xmodel(self)
```
##### Deploy simulation data of the quantized model
```py
  def dump_rnn_outputs_by_timestep(self, inputs)
```
    inputs: Input data that feed in the quantized model.  
