
## Versions supported

Python version 3.7 

TensorFlow 2.0

## Limitation

Only forward directional RNN networks using keras.layers.LSTMCell are supported.

Only support TensorFlow 2.0 Keras format input.

Only support quantization in TensorFlow 2.0 eager mode. 

## Install

Installation with Anaconda is recommened.

To install the nndct, do as follows:

#### Pre step 1 : install tensorflow 2.0 gpu version
    pip install tensorflow_gpu==2.0.0
    export CUDA_HOME with your cuda include folder in .bashrc 

#### Pre step 2 : compile GPU accleration kernel
    cd ..
    mkdir build
    cd build
    cmake ..
    make -j10
    cd ./tensorflow

#### Now install the main component:
    python setup.py install 

To create deployed model, XIR library needs to be installed. If just run quantization and check the accuracy, this is not must.

## Tool Usage

This chapter introduce using execution tools and APIs to implement quantization and generated model to be deployed on target hardware.  The APIs are in module nndct/tensorflow/tf_nndct/quantization/api.py:
#### Function tf_quantizer will create a quantizer.
```py
    def tf_quantizer(model,
                     input_signature,
                     quant_mode: int = 0,
                     output_dir: str = "quantize_result",
                     bitwidth_w: int = 8,
                     bitwidth_a: int = 8):
```
    model: Float module to be quantized.
    input_signature: input tensor with the same shape as real input of float module to be quantized, but the values can be random number.
    quant_mode: An integer that indicates which quantization mode the process is using. 0 for turning off quantization. 1 for calibration of quantization. 2 for evaluation of quantized model.
    output_dir: Directory for quantization result and intermediate files. Default is “quantize_result”.
    bitwidth_w: Global weights and bias quantization bit width. Default is 8.
    bitwidth_a: Global activation quantization bit width. Default is 8.
    
#### Example code
```py
    quantizer = tf_quantizer(......)
    model = quantizer.quant_model

    # Forwarding iterations of model, this part of code should share evaluation code of float model
    ...

    # output quantization result file(quant_info.json) in output directory
    quantizer.export_quant_config()

    # create xmodel for deployment in output directory
    quantizer.dump_xmodel()
```
