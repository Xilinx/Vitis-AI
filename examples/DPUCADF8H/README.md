## Getting Started to prepare and deploy a trained tensorflow model for FPGA acceleration

### Running tensorflow Benchmark Models
This directory provides scripts for running several well known models on the FPGA.

### Setup
> **Note:** Skip, If you have already run the below steps.

  Activate Conda Environment
  ```sh
  conda activate vitis-ai-tensorflow
  ```

  Setup the Environment

  ```sh
  # Typically, <path-to-vitis-ai> is `/workspace`
  source <path-to-vitis-ai>/setup/alveo/u200_u250/overlaybins/setup.sh
  ```

   Download a minimal validation set for [Imagenet2012](http://www.image-net.org/challenges/LSVRC/2012) using [Collective Knowledge (CK)](https://github.com/ctuning).
   > **Note:** User is responsible for the use of the downloaded content and compliance with any copyright licenses.

   ```sh
   python -m ck pull repo:ck-env
   python -m ck install package:imagenet-2012-val-min
   python -m ck install package:imagenet-2012-aux
   head -n 500 ~/CK-TOOLS/dataset-imagenet-ilsvrc2012-aux/val.txt > ~/CK-TOOLS/dataset-imagenet-ilsvrc2012-val-min/val.txt
   ```


### End-to-End Classification

  After the setup, run through a sample end-to-end tensorflow classification example using the following steps that demonstrates preparing and deploying a trained tensorflow model for FPGA acceleration using Xilinx Vitis-AI.

  The following example uses resnet50 tensorflow model from AI-Model-Zoo.

   Download Samples Models from [AI-Model-Zoo](../../models/AI-Model-Zoo)

  ```sh
  cd ${VAI_ALVEO_ROOT}/DPUCADF8H/tensorflow
  ```
#### Inspect TensorFlow model

   Inspect the tensorflow model to get input and output node(s), and input node shape.
   ```sh
   ./inspect_tf_model.sh <tensorflow_model>
   ```

   ```sh
   ./inspect_tf_model.sh models/resnet_v1_50_inference.pb
   ```

#### Quantize Model and Convert Model to XIR Model

  To deploy a tensorflow model on the FPGA, it needs to be quantized.

  *Quantize the model*

  The quantizer will generate scaling parameters for quantizing `float` to `INT8`. FPGAs take advantage of Fixed Point Precision to achieve more parallelization at lower power.

  ```sh
  vai_q_tensorflow quantize \
        --input_frozen_graph ./models/resnet_v1_50_inference.pb \
        --input_nodes input \
        --output_nodes resnet_v1_50/predictions/Reshape_1 \
        --input_fn utils.input_fn_resnet50_v1_tf \
        --input_shapes ?,224,224,3 \
        --calib_iter 10
  ```
  By default, the quantization result will be saved to `quaneize_results` directory under current work directory.

  *Convert the model to XIR format*
  ```sh
  xnnc-run \
        --type tensorflow \
        --layout NHWC \
        --model quantize_results/quantize_eval_model.pb \
        --out resnet50_v1_tf.xmodel \
        --inputs-shape 4,224,224,3
  ```

#### Compile the Model


  *Compile the Model*

  In this step, the network graph (xmodel file) is compiled using vitis-ai compiler.

  ```sh
  xcompiler -i ./resnet50_v1_tf.xmodel -o ./resnet50_v1_tf_compiled.xmodel -t DPUCADF8H 
  ```

#### Run example classification code

  This example code is modified from [VART demos](../../demo/VART).

  *Compile the executable*
  ```sh
  ./build.sh
  ```
  
  *Prepare test images*
  Prepare images and put it in a directory. e.g. `./images`

  *Run*
  ```sh
  ./resnet_example ./resnet50_v1_tf_compiled.xmodel ./images/
  ```
