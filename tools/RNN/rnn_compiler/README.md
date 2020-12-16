## RNN Compiler
RNN compiler compiles quantized RNN models and generate instructions for hardware. Currently RNN compiler only supports standard LSTM models.
RNN compiler contains two parts as follows:<br>
1. RNN compiler: compile LSTM models to hardward instructions.<br>
2. RNN simulator: simulate the process of LSTM models using computation graph and input data, get all the intermediate and final results of the model.<br>

## Supported and Limitation

Supports version 3.6 ~ 3.7.

Only supports Standard LSTM Models.

XIR library is needed.

## Quick Start in Docker environment

If you work in Vitis-AI 1.3 docker, there is a conda environment "vitis-ai-lstm", in which RNN compiler is already installed.
In this conda environment, python version is 3.6. You can directly start lstm example without installation steps.
- Copy examples/lstm_compiler_test.py to docker environment.
- Do the Compilation process, using the generated xmodel files.
  ```shell
  python lstm_compiler_test.py --model_path quantize_result/xmodel --device u50
  ```

## Install from source code

Installation with Anaconda is suggested. And if there is an old version of RNN compiler in the conda enviorment, suggest you remove all of its related files before install the new version. 

To install RNN compiler, do as follows:

#### Install the dependencies:
    pip install -r requirements.txt 

#### Install the main component:
    python setup.py install 

#### Verify the installation:
If the following command line does not report error, the installation is done.

    python -c "import lstm_compiler"

## RNN compiler Tool Usage

We provide simplest APIs to introduce our FPAG-friendly compilation feature. For a well-defined model, user only need to add 2-3 lines to do compilation process. An example is available in example/lstm_compiler_test.py.

#### Creating compilation script
1. Import the RNN compiler module <br>
   ```py
    from lstm_compiler.module import LstmModuleCompiler
   ```
2. Generate a compier object with Xmodel files path (this path should be the directory which contains Xmodel files, not the file path) and device to deploy(support U25 and U50 by now). <br>
   ```py
    compiler = LstmModuleCompiler(xmodels=args.model_path, device=args.device)
   ```
3. Do the compilation process. <br>
   ```py
    compiler.compile()
   ```

#### Run and output results
1. Run command to compiler model. <br>
   ```py
    python lstm_compiler_test.py --model_path quantize_result/xmodel --device u50
   ```
2. After running the commands, two subdirectories are generated. Computation graphs are stored in the subdirectory “CompileData", and instructions to deploy are stored in subdirectory “Instructions”. <br>

## RNN compiler main APIs

We provide RNN compiler APIs in file [lstm_compiler/modules.py](lstm_compiler/modules.py).

#### Class LstmModuleCompiler will create a RNN compiler object.
```py
class LstmModuleCompiler():
   def __init__(self, xmodels, device='u50')
```
    xmodels: directory which contains Xmodel, or list which contains Xmodels of the RNN model.
    device: device to deploy the RNN model. Default is 'u50'.

#### Class LstmModuleCompiler will create a RNN compiler object.
```py
def compile(self)
```
