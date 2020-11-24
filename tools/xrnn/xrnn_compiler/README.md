## XRNN Compiler
XRNN compiler compiles quantized RNN models and generate instructions for hardware. Currently XRNN compiler only support standard LSTM models. 
XRNN compiler contains two parts as follows:
1. XRNN compiler: compile LSTM models to hardward instructions.<br>
2. XRNN simulator: simulate the process of LSTM models using computation graph and input data, get all the intermediate and final results of the model.<br>

## Supported

Python version 3.6 ~ 3.7

XIR library

## Install

To install XRNN compiler as follows:

```py
pip install -r requirements.txt
python setup.py install
```

## Tool Usage

XRNN compiler provides LSTM compiler and LSTM simulator APIs, which are in file lstm_compiler/modules.py and lstm_simulator/module_simulator.py, respectively.

1. Class LstmModuleCompiler for LSTM compiler

```py
def __init__(self, xmodels)
```
Create one LstmModuleCompiler object. <br>
Xmodels: directory which contains xmodel files of the LSTM model, or list which contains xmodels of the LSTM model.
```py
def compile(self)
```
Compile LSTM xmodels to hardward instructions.

2. Class LstmModuleSimulator for LSTM simulator

```py
def __init__(self, data_path)
```
Create one LstmModuleSimulator object. <br>
data_path: directory for simulator data, which includes lstm quantizer dump data, compiler dump data and xmodel files.
```py
def run(self, data_check = False)
```
Run simulation process of the LSTM model.<br>
data_check: flag to control whether to check simulator dump data and quantizer dump data. Default is false.

## Tools
There are two tools in directory './examples', one for LSTM compiler, another for LSTM simulator.

1. For LSTM compiler:

```py
cd examples
python lstm_compiler_test.py --model_path [xmodel_path] --device [device_name]
```
xmodel_path: directory which contains xmodel files of the LSTM model layers.<br>
device_name: the device the generated instructions to load. Support U25 and U50 card by now.
Computation graphs are stored in the generated directory ./CompileData, and generated instructions are stored in the generated directory ./Instructions.

2. For LSTM simulator:

LSTM simulator must be run after LSTM compiler.<br>

Firstly, copy the compiler computation graphs data to the directory which contains quantizer dump data and xmodel files, as follows:
```py
cd examples
cp -ar CompilerData/* $quantizer_data_path
```

Secondly, run script as follows:
```py
python lstm_simulator_test.py --data_path [data_path] [--data_check] 
```
data_path: directory which contains quantizer dump data, compiler dump data and xmodel.<br>
data_check: flag to control whether to do data check after simulation. Default is false.
