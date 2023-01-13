# VAI_Q_PYTORCH

### Introduction
vai_q_pytorch is short for Vitis AI Quantizer for Pytorch. It is a tool for neural network model optimization with Pytorch model input.
vai_q_pytorch is designed as a part of a standard platform for neural network deep compression. Base on this architecture, working stages of vai_q_pytorch follows:<br>
    1.  Parse neural network computation graph from Pytorch framework to Intermediate Representation (IR).<br>
    2.  Modify the computation graph based on IR. Pruning, quantization and graph optimization are handled in this stage.<br>
    3.  Deploy the modified IR on different platform, such as DPU and Xilinx AI Engine.<br>
    4.  Assemble the modified computation graph back to Pytorch. In that way abilities and tools in Pytorch such as pre-processing, post processing and distribution system can be used.<br>


### Supported and Limitation

* Python
1. Support version 3.6 ~ 3.9. 
    
* Pytorch
1. Support version 1.1 ~ 1.12. 
2. QAT does NOT work with pytorch 1.1 ~ 1.3.
3. Data Parallelism is NOT supported.
    
* Models
1. Classification Models in Torchvision. 
2. Pytorch models in Xilinx Modelzoo. 
3. LSTM models (standard LSTM and customized LSTM)

### Quick Start in Docker environment

If you work in Vitis-AI 3.0 version of docker, there is a conda environment "vitis-ai-pytorch", in which vai_q_pytorch package is already installed. 
In this conda environment, python version is 3.7, pytorch version is 1.12 and torchvision version is 0.13. You can directly start our "resnet18" example without installation steps.
A new Conda environment with a specified PyTorch version (1.2~1.12) can be created using the script [replace_pytorch.sh](https://github.com/Xilinx/Vitis-AI/blob/master/docker/common/replace_pytorch.sh). This script clones a Conda environment from vitis-ai-pytorch, uninstalls the original PyTorch, Torchvision and vai_q_pytorch
packages, and then installs the specified version of PyTorch, Torchvision, and re-installs vai_q_pytorch from source code.
- Copy example/resnet18_quant.py to docker environment
- Download pre-trained [Resnet18 model](https://download.pytorch.org/models/resnet18-5c106cde.pth)
  ```shell
  wget https://download.pytorch.org/models/resnet18-5c106cde.pth -O resnet18.pth
  ```
- Prepare Imagenet validation images. [Pytorch example repo](https://github.com/pytorch/examples/tree/master/imagenet) for reference.
- Modify default data_dir and model_dir in resnet18_quant.py
- [Optional] Evaluate float model
  ```shell
  python resnet18_quant.py --quant_mode float
  ```
- [Optional] Inspect float model
  ```shell
  python resnet18_quant.py --quant_mode float --inspect --target DPUCAHX8L_ISA0_SP
  ```
- Quantize, using a subset (200 images) of validation data for calibration. Because we are in quantize calibration process, the displayed loss and accuracy are meaningless.
  ```shell
  python resnet18_quant.py --quant_mode calib --subset_len 200
  
  ```
- Evaluate quantized model
  ```shell
  python resnet18_quant.py --quant_mode test
  ```
- Export xmodel(or onnx)
  ```shell
  python resnet18_quant.py --quant_mode test --subset_len 1 --batch_size 1 --deploy
  ```
  
### Install from source code

Installation with Anaconda is suggested. And if there is an old version of vai_q_pytorch in the conda enviorment, suggest you remove all its related files before install the new version. 

To install vai_q_pytorch, do as follows:

##### Pre step 1: CUDA_HOME environment variable in .bashrc
For GPU version, if CUDA library is installed in /usr/local/cuda, add the following line into .bashrc. If CUDA is in other directory, change the line accordingly.

    export CUDA_HOME=/usr/local/cuda 

For CPU version, remove all CUDA_HOME environment variable setting in your .bashrc. Also it recommends to cleanup it in command line of a shell window with:

    unset CUDA_HOME

##### Pre step 2: install Pytorch(1.1-1.12.1) and torchvision
Here take pytorch 1.7.1 and torchvision 0.8.2 as an example, detailed instructions for other versions are in [pytorch](https://pytorch.org/) website.

    pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 -f https://download.pytorch.org/whl/torch_stable.html

##### Pre step 3: install other dependencies
    pip install -r requirements.txt 

##### Now install the main component:
For user NOT change the code frequently, install with commands:

    cd ./pytorch_binding; python setup.py install 
    
For user WILL change the code frequently, install with commands:

    cd ./pytorch_binding; python setup.py develop 

##### Verify the installation:
If the following command line does not report error, the installation is done.

    python -c "import pytorch_nndct"

To create deployed model, XIR library needs to be installed. If just run quantization and check the accuracy, this is not must. 
Refer to Vitis AI document for more information on deployment.

**Note:**<br>
If pytorch version you installed < 1.4, import pytorch_nndct before torch in your script. This is cuased by a pytorch bug before version 1.4.
Refer to Pytorch github issue [28536](https://github.com/pytorch/pytorch/pull/28536) and [19668](https://github.com/pytorch/pytorch/issues/19668) for details. 
```python
import pytorch_nndct
import torch
```

### vai_q_pytorch Tool Usage

vai_q_pytorch is designed to work as a Pytorch plugin. We provide simplest APIs to introduce our FPGA-friendly quantization feature.
For a well-defined model, user only need to add 2-3 lines to get a quantize model object.

##### Model pre-requirements for quantizer
- The model to be quantized should include forward method only. All other functions should be moved outside or move to a derived class. 
These functions usually work as pre-processing and post-processing. If they are not moved outside, 
our API will remove them in our quantized module, which will cause unexpected behavior when forwarding quantized module. <br>
- The float model should pass "jit trace test". First set the float module to evaluation status, then use “torch.jit.trace” function to test the float model. Make sure the float module can pass the trace test.For more details, please refer to example/jupyter_notebook/jit_trace_test/jit_trace_test.ipynb.<br>
- The most common operators in pytorch are supported in quantizer, please refer to [support_op.md](doc/support_op.md) for details.
##### Inspect float model before quantization
Vai_q_pytorch provides a function called inspector to help users diagnose neural network (NN) models under different device architectures. The inspector can predict target device assignments based on hardware constraints.The generated inspection report can be used to guide  users to modify or optimize the NN model, greatly reducing the difficulty and time of deployment. It is recommended to inspect float models before quantization.

Take resnet18_quant.py to demonstrate how to apply this feature.
1. Import vai_q_pytorch modules <br>
   ```py
    from pytorch_nndct.apis import Inspector
   ```
2. Create a inspector with target name or fingerprint. <br>
   ```py
    inspector = Inspector(target) 
   ```
3. Inspect float model. <br>
   ```py
    input = torch.randn([batch_size, 3, 224, 224])
    inspector.inspect(model, input)
   ```
Run command with "--quant_mode float&emsp;--inspect&emsp;--target {target_name}" to inspect model.
```py
    python resnet18_quant.py --quant_mode float --inspect --target DPUCAHX8L_ISA0_SP
```
Inspector will display some special messages on screen with special color and special keyword prefix "VAIQ_*" according to the verbose_level setting.Note the messages displayed between "[VAIQ_NOTE]: =>Start to inspect model..." and "[VAIQ_NOTE]: =>Finish inspecting."

If the inspector runs successfully, three important files are usually generated under the output directory "./quantize_result".
```
    inspect_{target}.txt: Target information and all the details of operations in float model.
    inspect_{target}.svg: If image_format is not None. A visualization of inspection result is generated.
    inspect_{target}.gv: If image_format is not None. Dot source code of inspetion result is generated. 
```
Note:
* The inspector relies on 'xcompiler' package. In conda env vitis-ai-pytorch in Vitis-AI docker, xcompiler is ready. But if vai_q_pytorch is installed by source code, it needs to install xcompiler in advance.<br>
* Visualization of inspection results relies on the dot engine.If you don't install dot successfully, set 'image_format = None' when inspecting.
* If you need more detailed guidance, you can refer to ./example/jupyter_notebook/inspector/inspector_tutorial.ipynb. Please install jupyter notebook in advance. Run the following command:
```py
jupyter notebook example/jupyter_notebook/inspector/inspector_tutorial.ipynb
```


##### Add vai_q_pytorch APIs to float scripts
Before quantization, suppose there is a trained float model and some python scripts to evaluate model's accuracy/mAP. 
Quantizer API will replace float module with quantized module and normal evaluate function will encourage quantized module forwarding. 
Quantize calibration determines "quantize" op parameters in evaluation process if we set flag quant_mode to "calib". 
After calibration, we can evaluate quantized model by setting quant_mode to "test".

Take resnet18_quant.py to demonstrate how to add vai_q_pytorch APIs in float code. 
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
   ```py
    acc1_gen, acc5_gen, loss_gen = evaluate(quant_model, val_loader, loss_fn)
   ```
4. Output quantization result and deploy model. <br>
   ```py
   if quant_mode == 'calib':
     quantizer.export_quant_config()
   if deploy:
     quantizer.export_torch_script()
     quantizer.export_onnx_model()
     quantizer.export_xmodel()
   ```

##### Run and output results
Before running commands, let's introduce the log message in vai_q_pytorch. vai_q_pytorch log messages have special color and special keyword prefix "VAIQ_*". 
vai_q_pytorch log message types include "error", "warning" and "note". 
Pay attention to vai_q_pytorch log messages to check the flow status.<br>
* Run command with "--quant_mode calib" to quantize model.
```py
    python resnet18_quant.py --quant_mode calib --subset_len 200
```
When doing calibration forward, we borrow float evaluation flow to minimize code change from float script. So there are loss and accuracy displayed in the end. 
They are meaningless, just skip them. Pay more attention to the colorful log messages with special keywords "VAIQ_*".

Another important thing is to control iteration numbers during quantization and evaluation. 
Generally, 100-1000 images are enough for quantization and the whole validation set are required for evaluation. 
The iteration numbers can be controlled in the data loading part.
In this case, argument "subset_len" controls how many images used for network forwarding. 
But if the float evaluation script doesn't have an argument with similar role, it is better to add one, otherwise it should be changed manually.

If this quantization command runs successfully, two important files will be generated under output directory “./quantize_result”. 
```
    ResNet.py: converted vai_q_pytorch format model, 
    Quant_info.json: quantization steps of tensors got. (Keep it for evaluation of quantized model)
```
* To evaluate quantized model, run the following command:
```shell
    python resnet18_quant.py --quant_mode test 
```
When this command finishes, the displayed accuracy is the right accuracy for quantized model. <br> 

* To export xmodel, batch size 1 is must for compilation, and subset_len=1 is to avoid redundant iteration. Run the following command:
```shell
    python resnet18_quant.py --quant_mode test --subset_len 1 --batch_size 1 --deploy
```
Skip loss and accuracy displayed in log in this run. 
Xmodel file for Vitis AI compiler will be generated under output directory “./quantize_result”. It will be further used to deploy this model to FPGA. 
```
    ResNet_int.xmodel: deployed XIR format model
    ResNet_int.onnx:   deployed onnx format model
    ResNet_int.pt:     deployed torch script format model
```
In conda env vitis-ai-pytorch in Vitis-AI docker, XIR is ready. But if vai_q_pytorch is installed by source code, it needs to install XIR in advance.<br>
If XIR is not installed, xmodel file can't be generated, this command will raise error in the end.


##### Module partial quantization
Sometimes not all the sub-modules in a module will be quantized. Besides call general vai_q_pytorch APIS, QuantStub/DeQuantStub OP pair can be used to realize it.<br>
Example code for quantizing subm0 and subm2, but not to quantize subm1: 
```py
from pytorch_nndct.nn import QuantStub, DeQuantStub

class WholeModule(torch.nn.module):
    def __init__(self,...):
        self.subm0 = ...
        self.subm1 = ...
        self.subm2 = ...

        # define QuantStub/DeQuantStub submodules
        self.quant = QuantStub()
        self.dequant = DeQuantStub()

    def forward(self, input):
        input = self.quant(input) # begin of part to be quantized
        output0 = self.subm0(input)
        output0 = self.dequant(output0) # end of part to be quantized

        output1 = self.subm1(output0)

        output1 = self.quant(output1) # begin of part to be quantized
        output2 = self.subm2(output1)
        output2 = self.dequant(output2) # end of part to be quantized
```
##### Register Custom Operation
In the XIR Op library, there's a well-defined set of operators to cover the wildly used deep learning frameworks, e.g. TensorFlow, Pytorch and Caffe, and all of the build-in operators for DPU. This enhanced the expression ability and achieved one of the core goals, which is eliminating the difference between these frameworks and providing a unified representation for users and developers. However, this Op library can’t cover all of Ops in the upstream frameworks. So, the XIR provide a new op definition to express all other unknown Ops for Op library. In order to convert a quantized model to xmodel，vai_q_pytorch provide a decorator to register a operation or a group of operations as a custom operation which is unknown for XIR.
```py
# Decorator API
def register_custom_op(op_type: str, attrs_list: Optional[List[str]] = None):
  """The decorator is used to register the function as a custom operation.
  Args:
  op_type(str) - the operator type registered into quantizer. 
                 The type should not conflict with pytorch_nndct
                
  attrs_list(Optional[List[str]], optional) - 
  the name list of attributes that define operation flavor. 
  For example, Convolution operation has such attributes as padding, dilation, stride and groups. 
  The order of name in attrs_list should be consistent with that of the arguments list. 
  Default: None
  
  """
```
How to use: <br>
-	Aggregate some operations as a function. The first argument name of this function should be ctx. The meaning of ctx is the same as that in torch.autograd.Function.
-	Decorate this function with the decorator described above. 
-	Example model code is in example/resnet18_quant_custom_op.py. On how to run it, please refer to example/resnet18_quant.py.
```py
from pytorch_nndct.utils import register_custom_op

@register_custom_op(op_type="MyOp", attrs_list=["scale_1", "scale_2"])
def custom_op(ctx, x: torch.Tensor, y:torch.Tensor, scale_1:float, scale_2:float) -> torch.Tensor:
	return scale_1 * x + scale_2 * y  

class MyModule(torch.nn.Module):
def __init__(self):
   ...

def forward(self, x, y):
   return custom_op(x, y, 0.5, 0.5)
```

Limitations: <br>
-	Loop operation is not allowed in a custom operation
-	The number of return values for a custom operation can only be one.

##### Fast finetune model 
Sometimes direct quantization accuracy is not high enough, then it needs finetune model parameters. <br>
- The fast finetuning is not real training of the model, and only needs limited number of iterations. For classification models on Imagenet dataset, 5120 images are enough in general.
- It only needs do some modification based on evaluation model script and does not need setup optimizer for training.
- A function for model forwarding iteration is needed and will be called among fast finetuning. 
- Re-calibration with original inference code is highly recommended.
- Example code in example/resnet18_quant.py:
```py
# fast finetune model or load finetuned parameter before test 
  if finetune == True:
      ft_loader, _ = load_data(
          subset_len=5120,
          train=False,
          batch_size=batch_size,
          sample_method='random',
          data_dir=args.data_dir,
          model_name=model_name)
      if quant_mode == 'calib':
        quantizer.fast_finetune(evaluate, (quant_model, ft_loader, loss_fn))
      elif quant_mode == 'test':
        quantizer.load_ft_param()
```
- For example/resnet18_quant.py, command line to do parameter fast finetuning and re-calibration,
  ```shell
  python resnet18_quant.py --quant_mode calib --fast_finetune
  
  ```
- command line to test fast finetuned quantized model accuracy,
  ```shell
  python resnet18_quant.py --quant_mode test --fast_finetune

  ```
- command line to deploy fast finetuned quantized model,
  ```shell
  python resnet18_quant.py --quant_mode test --fast_finetune --subset_len 1 --batch_size 1 --deploy

  ```

##### Finetune quantized model
- This mode can be used to finetune a quantized model (loading float model parameters), also can be used to do quantization-aware-training (QAT) from scratch.
- It needs to add some vai_q_pytorch interface functions based on the float model training script.
- The mode requests the trained model cannot use +/- operator in model forwarding code. It needs to replace them with torch.add/torch.sub module.
- Example code in example/resnet18_qat.py:
```py
  # create quantizer can do QAT
  input = torch.randn([batch_size, 3, 224, 224], dtype=torch.float32)
  from pytorch_nndct import QatProcessor
  qat_processor = QatProcessor(model, inputs, bitwidth=8)
  quantized_model = qat_processor.trainable_model()
```
```py
  # get the deployable model and test it
  output_dir = 'qat_result'
  deployable_model = qat_processor.to_deployable(quantized_model, output_dir)
  validate(val_loader, deployable_model, criterion, gpu)
```
```py
  # export xmodel from deployable model
  # need at least 1 iteration of inference with batch_size=1 
  # Use cpu mode to export xmodel.
  deployable_model.cpu()
  val_subset = torch.utils.data.Subset(val_dataset, list(range(1)))
  subset_loader = torch.utils.data.DataLoader(
    val_subset,
    batch_size=1,
    shuffle=False,
    num_workers=8,
    pin_memory=True)
  # Must forward deployable model at least 1 iteration with batch_size=1
  for images, _ in subset_loader:
    deployable_model(images)
  qat_processor.export_xmodel(output_dir)
```

##### Configuration of Quantization Strategy
For multiple quantization strategy configurations, vai_q_pytorch supports quantization configuration file in JSON format. 
And we only need to pass the configuration file to torch_quantizer API. Example code in resnet18_quant.py:
```py
config_file = "./pytorch_quantize_config.json"
quantizer = torch_quantizer(quant_mode=quant_mode, 
                            module=model, 
                            input_args=(input), 
                            device=device, 
                            quant_config_file=config_file)
```
For detailed information of the json file contents, please refer to [Quant_Config.md](doc/Quant_Config.md)

##### Hardware-Aware Quantization Strategy
The Inspector provides device assignments to operators in the neural network based on the target device. vai_q_pytorch can use the power of inspector to perform hardware-aware quantization.
Example code in resnet18_quant.py:
```py
quantizer = torch_quantizer(quant_mode=quant_mode, 
                            module=model, 
                            input_args=(input), 
                            device=device, 
                            quant_config_file=config_file, target=target)
```
- For example/resnet18_quant.py, command line to do hardware-aware calibration,
  ```shell
  python resnet18_quant.py --quant_mode calib --target DPUCAHX8L_ISA0_SP
  
  ```
- command line to test hardware-aware quantized model accuracy,
  ```shell
  python resnet18_quant.py --quant_mode test --target DPUCAHX8L_ISA0_SP

  ```
- command line to deploy quantized model,
  ```shell
  python resnet18_quant.py --quant_mode test --target DPUCAHX8L_ISA0_SP --subset_len 1 --batch_size 1 --deploy

  ```
### vai_q_pytorch main APIs

The APIs for CNN are in module [pytorch_binding/pytorch_nndct/apis.py](pytorch_binding/pytorch_nndct/apis.py):
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
               device: torch.device = torch.device("cuda"),
               quant_config_file: Optional[str] = None,
               target: Optional[str]=None):

```
    quant_mode: An integer that indicates which quantization mode the process is using. "calib" for calibration of quantization. "test" for evaluation of quantized model.
    Module: Float module to be quantized.
    Input_args: Input tensor with the same shape as real input of float module to be quantized, but the values can be random number.
    State_dict_file: Float module pretrained parameters file. If float module has read parameters in, the parameter is not needed to be set.
    Output_dir: Directory for quantization result and intermediate files. Default is “quantize_result”.
    Bitwidth: Global quantization bit width. Default is 8.
    Device: Run model on GPU or CPU.
    quant_config_file: Json file path for quantization strategy configuration
    target: If target device is specified, the hardware-aware quantization is on. Default is None.
    
##### Export quantization steps information
```py
  def export_quant_config(self):
```
##### Export xmodel and dump OPs output data for detailed data comparison
```py
  def export_xmodel(self, output_dir, deploy_check):
```
    Output_dir: Directory for quantization result and intermediate files. Default is “quantize_result”
    Depoly_check: Flags to control dump of data for detailed data comparison. Default is False. If it is set to True, binary format data will be dumped to output_dir/deploy_check_data_int/.
##### Export onnx format quantized model
```py
  def export_onnx_model(self, output_dir, verbose):
```
    Output_dir: Directory for quantization result and intermediate files. Default is “quantize_result”
    Verbose: Flag to control showing verbose log or no
##### Export torchscript format quantized model
```py
  def export_torch_script(self, output_dir="quantize_result", verbose=False):
```
    Output_dir: Directory for quantization result and intermediate files. Default is “quantize_result”
    Verbose: Flag to control showing verbose log or not

##### Create a inspector
```py
class Inspector():
  def __init__(self, name_or_fingerprint: str):
```
    name_or_fingerprint: Specify the hardware target name or fingerprint
##### Inspect float model
```py
  def inspect(self, 
              module: torch.nn.Module, 
              input_args: Union[torch.Tensor, Tuple[Any]], 
              device: torch.device = torch.device("cuda"),
              output_dir: str = "quantize_result",
              verbose_level: int = 1,
              image_format: Optional[str] = None):
```
    module: Float module to be depolyed
    input_args: Input tensor with the same shape as real input of float module, but the values can be random number.    
    device: Trace model on GPU or CPU.
    output_dir: Directory for inspection results
    verbose_level: Control the level of detail of the inspection results displayed on the screen. Defaut:1
        0: turn off printing inspection results.
        1: print summary report of operations assigned to CPU.
        2: print summary report of device allocation of all operations.
    image_format: Export visualized inspection result. Support 'svg' / 'png' image format.



