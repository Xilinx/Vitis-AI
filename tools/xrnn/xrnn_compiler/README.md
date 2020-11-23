DCTC_LSTM is deep learning compiler toolchain for LSTM. By now, it has contained two functions as follows:

1. LSTM compiler: compile LSTM xmodels to hardward instructions.<br>
2. LSTM simulator: simulate the process of LSTM models using computation graph and input data, get all the intermediate and final results of the model.<br>

 ## Supported

Python version 3.6 ~ 3.7

XIR library

## Install

To install the dctc_lstm as follows:

```py
pip install -r requirements.txt
python setup.py install
```
PREFIX: directory to store the dynamic link libraries.

## Tool Usage

Dctc_lstm provides LSTM compiler and LSTM simulator APIs, which are in file lstm_compiler/modules.py and lstm_simulator/module_simulator.py, respectively.

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
data_path: directory for simulator data, which includes nndct dump data, compiler dump data and xmodel.
```py
def run(self, data_check = False)
```
Run simulation process of the LSTM model.<br>
data_check: flag to control whether to check simulator dump data and nndct dump data. Default is false.

## Tools
There are two tools in directory './examples', one for LSTM compiler, another for LSTM simulator.

1. For LSTM compiler:

```py
cd examples
python lstm_compiler_test.py --model_path [xmodel_path] --device [device_name]
```
xmodel_path: directory which contains xmodel files of the LSTM model layers.<br>
device_name: the device the generated instructions to load. Support u25 and u50 by now.
Computation graphs are stored in the generated directory ./CompileData, and generated instructions are stored in the generated directory ./Instructions.

2. For LSTM simulator:

LSTM simulator must be run after LSTM compiler.<br>

Firstly, copy the compiler computation graphs data to the directory which contains nndct dump data and xmodel files, as follows:
```py
cd examples
cp -ar CompilerData/* $nndct_data_path
```

Secondly, run script as follows:
```py
python lstm_simulator_test.py --data_path [data_path] [--data_check] 
```
data_path: directory which contains nndct dump data, compiler dump data and xmodel.<br>
data_check: flag to control whether to do data check after simulation. Default is false.
