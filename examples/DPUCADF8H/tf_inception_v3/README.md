## Getting Started to prepare and deploy a trained Tensorflow 1.x model for FPGA acceleration

### Running Tensorflow Benchmark Models
This directory provides scripts for running several well known models on the FPGA.

### Setup
> **Note:** Skip, If you have already run the below steps.

#### Alveo Card Setup in Host
  Please refer to [Alveo Card Setup Instructions](/setup/alveo#alveo-card-setup-in-host) for Alveo card setup in host.

#### Environment Variable Setup in Docker Container
  Activate Conda Environment
  ```sh
  conda activate vitis-ai-tensorflow
  ```

  DPU IP selection

  ```sh
  # Typically, <path-to-vitis-ai> is `/workspace`
  source <path-to-vitis-ai>/setup/alveo/setup.sh DPUCADF8H
  ```

   Download a minimal validation set for [Imagenet2012](http://www.image-net.org/challenges/LSVRC/2012) using [Collective Knowledge (CK)](https://github.com/ctuning).
   > **Note:** User is responsible for the use of the downloaded content and compliance with any copyright licenses.

   ```sh
   python -m ck pull repo:ck-env
   python -m ck install package:imagenet-2012-val-min
   python -m ck install package:imagenet-2012-aux --tags=from.berkeley
   head -n 500 ~/CK-TOOLS/dataset-imagenet-ilsvrc2012-aux-from.berkeley/val.txt > ~/CK-TOOLS/dataset-imagenet-ilsvrc2012-val-min/val.txt
   ```


### End-to-End Classification

  After the setup, run through a sample end-to-end Tensorflow classification example using the following steps that demonstrates preparing and deploying a trained Tensorflow model for FPGA acceleration using Xilinx Vitis-AI.

  The following example uses the pretrained Inception-v3 Tensorflow model from the AI-Model-Zoo.

   Download sample models from [AI-Model-Zoo](../../../models/AI-Model-Zoo)

   In the following steps your downloaded model's .pb file should be placed inside a models directory

#### Inspect TensorFlow model

   Inspect the Tensorflow model to get input and output node(s), and input node shape.
   ```sh
   Usage: ./inspect_tf_model.sh <tensorflow_model>
   ```

   ```sh
   ./inspect_tf_model.sh models/inception_v3.pb
   ```

#### Quantize the Model

  To deploy a Tensorflow model on the FPGA, it needs to be quantized.

  *Quantize the model*

  The quantizer will generate scaling parameters for quantizing `float` to `INT8`. FPGAs take advantage of Fixed Point Precision to achieve more parallelization at lower power.

  ```sh
  vai_q_tensorflow quantize \
        --input_frozen_graph ./models/inception_v3.pb \
        --input_nodes input \
        --output_nodes InceptionV3/Predictions/Reshape_1 \
        --input_fn utils.input_fn_inception_v3_tf \
        --input_shapes ?,299,299,3 \
        --calib_iter 10
  ```
  By default, the quantization result will be saved to `quantize_results` directory under current work directory.

#### Compile the Model

  In this step, the network graph (xmodel file) is compiled using the Vitis-AI compiler.  Note this may take approximately 5 minutes.

  ```sh
  vai_c_tensorflow \
        --arch /opt/vitis_ai/compiler/arch/DPUCADF8H/U250/arch.json \
        --frozen_pb quantize_results/quantize_eval_model.pb \
        --output_dir out \
        --net_name tf_inception_v3_compiled \
        --options '{"input_shape": "4,299,299,3"}'
  ```
  > **Note:** DPUCADF8H uses a default batchsize of 4. If the original model's batchsize is 1, you will need to specify the `input_shape` using the `--options` argument as the command above. The `--options` is a general argument for vai\_c\_caffe/vai\_c\_tensorflow/vai\_c\_tensorflow2.

#### Run example classification code

  This example code is derived from [VART demos](../../../demo/VART).

  *Compile the executable*
  ```sh
  ./build.sh
  ```
  
  *Prepare test images*
  This demo will run on all images in the specified directory. e.g. `./images/`

  *Run*
  ```sh
  ./inception_example ./tf_inception_v3_compiled.xmodel ./images/
  ```
