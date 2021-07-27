# VAI_Q_PYTORCH

### Introduction
vai_q_pytorch is short for Vitis AI Quantizer for Pytorch. It is a tool for neural network model optimization with Pytorch model input.
vai_q_pytorch is designed as a part of a standard platform for neural network deep compression. Base on this architecture, working stages of vai_q_pytorch follows:<br>
    1.  Parse neural network computation graph from Pytorch framework to Intermediate Representation (IR).<br>
    2.  Modify the computation graph based on IR. Pruning, quantization and graph optimization are handled in this stage.<br>
    3.  Deploy the modified IR on different platform, such as DPU and Xilinx AI Engine.<br>
    4.  Assemble the modified computation graph back to Pytorch. In that way abilities and tools in Pytorch such as pre-processing, post processing and distribution system can be used.<br>


### Supported and Limitaion

* Python
1. Support version 3.6 ~ 3.7. 
    
* Pytorch
1. Support version 1.1 ~ 1.7.1. 
2. QAT does NOT work with pytorch 1.1.
3. Data Parallelism is NOT supported.
    
* Models
1. Classification Models in Torchvision. 
2. Pytorch models in Xilinx Modelzoo. 
3. LSTM models(standard LSTM and customized LSTM)

### Quick Start in Docker environment

If you work in Vitis-AI 1.3 and later veriosn of docker, there is a conda environment "vitis-ai-pytorch", in which vai_q_pytorch package is already installed. 
In this conda environment, python version is 3.6, pytorch version is 1.4.0 and torchvision version is 0.5.0. You can directly start our "resnet18" example without installation steps.
A new Conda environment with a specified PyTorch version (1.2~1.7.1) can be created using the /opt/vitis_ai/scripts/replace_pytorch.sh script. This script clones a Conda environment from vitis-ai-pytorch, uninstalls the original PyTorch, Torchvision and vai_q_pytorch
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
- Quantize, using a subset(200 images) of validation data for calibration. Because we are in quantize calibration process, the displayed loss and accuracy are meaningless.
  ```shell
  python resnet18_quant.py --quant_mode calib --subset_len 200
  
  ```
- Evaluate quantized model
  ```shell
  python resnet18_quant.py --quant_mode test
  ```
- Export xmodel 
  ```shell
  python resnet18_quant.py --quant_mode test --subset_len 1 --batch_size 1 --deploy
  ```
  
### Install from source code

Installation with Anaconda is suggested. And if there is an old version of vai_q_pytorch in the conda enviorment, suggest you remove all of its related files before install the new version. 

To install vai_q_pytorch, do as follows:

##### Pre step 1 : CUDA_HOME environment variable in .bashrc
For GPU version, if CUDA library is installed in /usr/local/cuda, add the following line into .bashrc. If CUDA is in other directory, change the line accordingly.

    export CUDA_HOME=/usr/local/cuda 

For CPU version, remove all CUDA_HOME environment variable setting in your .bashrc. Also it recommends to cleanup it in command line of a shell window with:

    unset CUDA_HOME

##### Pre step 2 : install Pytorch(1.1-1.7.1) and torchvision
Here take pytorch 1.7.1 and torchvision 0.8.2 as an example, detailed instructions for other versions are in [pytorch](https://pytorch.org/) website.

    pip install torch==1.7.1+cu101 torchvision==0.8.2+cu101 -f https://download.pytorch.org/whl/torch_stable.html

##### Pre step 3 : install other dependencies
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
our API will remove them in our quantized module, which will cause unexpected behaviour when forwarding quantized module. <br>
- The float model should pass "jit trace test". First set the float module to evaluation status, then use “torch.jit.trace” function to test the float model. Make sure the float module can pass the trace test. <br>

##### Add vai_q_pytorch APIs to float scripts
Before quantization, suppose there is a trained float model and some python scripts to evaluate model's accuracy/mAP. 
Quantizer API will replace float module with quantized module and normal evaluate function will encourage quantized module forwarding. 
Quantize calibration determines "quantize" op parameters in evaluation process if we set flag quant_mode to "calib". 
After calibration, we can evaluate quantized model by setting quant_mode to "test".

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
   ```py
    acc1_gen, acc5_gen, loss_gen = evaluate(quant_model, val_loader, loss_fn)
   ```
4. Output quantization result and deploy model. <br>
   ```py
   if quant_mode == 'calib':
     quantizer.export_quant_config()
   if deploy:
     quantizer.export_xmodel()
   ```

##### Run and output results
Before running commands, let's introduce the log message in vai_q_pytorch. vai_q_pytorch log messages have special color and special keyword "NNDCT". 
"NNDCT" is our internal project name and we will change it later. vai_q_pytorch log message types include "error", "warning" and "note". 
Pay attention to vai_q_pytorch log messages to check the flow status.<br>
* Run command with "--quant_mode calib" to quantize model.
```py
    python resnet18_quant.py --quant_mode calib --subset_len 200
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
    ResNet_int.xmodel: deployed model
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

##### Fast finetune model 
Sometimes direct quantization accuracy is not high enough, then it needs finetune model parameters. <br>
- The fast finetuning is not real training of the model, and only needs limited number of iterations. For classification models on Imagenet dataset, 1000 images are enough in general.
- It only needs do some modification based on evaluation model script and does not need setup optimizer for training.
- A function for model forwarding iteration is needed and will be called among fast finetuning. 
- Re-calibration with original inference code is highly recommended.
- Example code in example/resnet18_quant.py:
```py
# fast finetune model or load finetuned parameter before test 
  if finetune == True:
      ft_loader, _ = load_data(
          subset_len=1024,
          train=False,
          batch_size=batch_size,
          sample_method=None,
          data_dir=args.data_dir,
          model_name=model_name)
      if quant_mode == 'calib':
        quantizer.fast_finetune(evaluate, (quant_model, ft_loader, loss_fn))
      elif quant_mode == 'test':
        quantizer.load_ft_param()
```
- For example/resnet18_quant.py, command line to do parameter finetuning and re-calibration,
  ```shell
  python resnet18_quant.py --quant_mode calib --fast_finetune
  
  ```
- command line to test finetuned quantized model accuracy,
  ```shell
  python resnet18_quant.py --quant_mode test --fast_finetune

  ```

##### Finetune quantized model
- This mode can be used to finetune a quantized model(loading float model parameters), also can be used to do quantization-aware-training (QAT) from scratch.
- It needs to add some vai_q_pytorch interface functions based on the float model training script.
- The mode does not work with pytorch version 1.1. 
- The mode requests the trained model can not use +/- operator in model forwarding code. It needs to replace them with torch.add/torch.sub module.
- Example code in example/resnet18_qat.py:
```py
  # vai_q_pytorch interface function: create quantizer can do QAT
  input = torch.randn([batch_size, 3, 224, 224], dtype=torch.float32)
  quantizer = torch_quantizer(quant_mode = 'calib',
                              module = model, 
                              input_args = input,
                              bitwidth = 8,
                              mix_bit = False,
                              qat_proc = True)
  quantized_model = quantizer.quant_model
```
```py
    # vai_q_pytorch interface function: deploy the trained model and convert xmodel
    # need at least 1 iteration of inference with batch_size=1 
    quantizer.deploy(quantized_model)
    deployable_model = quantizer.deploy_model
    val_dataset2 = torch.utils.data.Subset(val_dataset, list(range(1)))
    val_loader2 = torch.utils.data.DataLoader(
        val_dataset2,
        batch_size=1,
        shuffle=False,
        num_workers=workers,
        pin_memory=True)
    validate(val_loader2, deployable_model, criterion, gpu)
    quantizer.export_xmodel()
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
               qat_proc: bool = False):

```
    quant_mode: An integer that indicates which quantization mode the process is using. "calib" for calibration of quantization. "test" for evaluation of quantized model.
    Module: Float module to be quantized.
    Input_args: Input tensor with the same shape as real input of float module to be quantized, but the values can be random number.
    State_dict_file: Float module pretrained parameters file. If float module has read parameters in, the parameter is not needed to be set.
    Output_dir: Directory for quantization result and intermediate files. Default is “quantize_result”.
    Bitwidth: Global quantization bit width. Default is 8.
    Device: Run model on GPU or CPU.
    Qat_proc: Turn on quantization-aware-training (QAT).
    
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




