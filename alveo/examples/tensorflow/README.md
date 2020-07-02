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
  source /workspace/alveo/overlaybins/setup.sh
  ```

   Download a minimal validation set for [Imagenet2012](http://www.image-net.org/challenges/LSVRC/2012) using [Collective Knowledge (CK)](https://github.com/ctuning).
   > **Note:** User is responsible for the use of the downloaded content and compliance with any copyright licenses.

   ```sh
   python -m ck pull repo:ck-env
   python -m ck install package:imagenet-2012-val-min
   python -m ck install package:imagenet-2012-aux
   head -n 500 ~/CK-TOOLS/dataset-imagenet-ilsvrc2012-aux/val.txt > ~/CK-TOOLS/dataset-imagenet-ilsvrc2012-val-min/val.txt
   ```

   Download Samples Models (will be downloaded into `./models` directory).

   ```sh
   cd ${VAI_ALVEO_ROOT}/examples/tensorflow
   python getModels.py
   ```

### End-to-End Classification

  After the setup, run through a sample end-to-end tensorflow classification example using the following steps that demonstrates preparing and deploying a trained tensorflow model for FPGA acceleration using Xilinx Vitis-AI.

  The following example uses the googlenet example. You can try the flow with the other models found in `${VAI_ALVEO_ROOT}/examples/tensorflow/models` directory.

  ```sh
  cd ${VAI_ALVEO_ROOT}/examples/tensorflow
  ```
#### Inspect TensorFlow model

   Inspect the tensorflow model to get input and output node(s), and input node shape. 
   ```sh
   ./inspect_tf_model.sh <tensorflow_model>
   ```
   
   ```sh
   ./inspect_tf_model.sh models/inception_v1_baseline.pb
   ```
   
#### Quantize Model

  To deploy a tensorflow model on the FPGA, it needs to be quantized.

  *Quantize the model* 
  
  The quantizer will generate scaling parameters for quantizing `float` to `INT8`. FPGAs take advantage of Fixed Point Precision to achieve more parallelization at lower power.

  The above step is invoked by using the `--quantize` switch. If you plan to work on several models, you can use the `--output_dir` switch to generate and store model artifacts in seperate directories. By default, output_dir is `./work`

  ```sh
  python run.py --quantize --model models/inception_v1_baseline.pb --pre_process inception_v1 --output_dir work --input_nodes data --output_nodes loss3_loss3 --input_shapes 1,224,224,3 --batch_size 16
  ```

#### Partition, Compile and Run Inference

  This performs compilation and subgraph partitioning in a single step. To run a tensorflow model on the FPGA, it needs to be compiled and a new graph needs to be generated. The new graph is similar to the original, with the FPGA subgraph removed, and replaced with a custom Python layer.

  *Compile the Model*
  
  In this step, the network graph (frozen `.pb` file) is compiled.

  *Subgraph Partitioning*
  
  In this step, the original graph is cut, and a custom FPGA accelerated python layer is inserted to be used for Inference.

  The above two steps are invoked by using the `--validate` switch. If you plan to work on several models, you can use the `--output_dir` switch to generate and store model artifacts in seperate directories. By default, output_dir is `./work`.

  ```sh
  python run.py --validate --model models/inception_v1_baseline.pb --pre_process inception_v1 --output_dir work --input_nodes data --output_nodes loss3_loss3 --c_input_nodes data --c_output_nodes pool5_7x7_s1 --input_shapes ?,224,224,3

  # Note: You might need to change run.py , and adjust the key name of the returned dictionary in the calib_input, preprocess_default and preprocess_inception functions. Adjust the key name to match the name of the input tensor to your graph.
  ```

 #### Other Models
 
 Below are the commands for running compilation, partitioning and inference with other models downloaded by `getModels.py`.

  ```sh
  ## resnet_50
  python run.py --quantize --model models/resnet50_baseline.pb --output_dir work --input_nodes data --output_nodes prob --input_shapes 1,224,224,3 --pre_process resnet50
  python run.py --validate --model models/resnet50_baseline.pb --output_dir work --input_nodes data --output_nodes prob --c_input_nodes data --c_output_nodes prob --input_shapes ?,224,224,3 --pre_process resnet50

  ## resnet_101
  python run.py --quantize --model models/resnet_v1_101.pb --output_dir work --input_nodes input --output_nodes resnet_v1_101/predictions/Softmax --input_shapes 1,224,224,3 --pre_process resnet_v1_101
  python run.py --validate --model models/resnet_v1_101.pb --output_dir work --input_nodes input --output_nodes resnet_v1_101/predictions/Softmax --c_input_nodes input --c_output_nodes resnet_v1_101/logits/BiasAdd --input_shapes ?,224,224,3 --pre_process resnet_v1_101

  ## resnet_152
  python run.py --quantize --model models/resnet_v1_152.pb --output_dir work --input_nodes input --output_nodes resnet_v1_152/predictions/Softmax --input_shapes 1,224,224,3 --pre_process resnet_v1_152
  python run.py --validate --model models/resnet_v1_152.pb --output_dir work --input_nodes input --output_nodes resnet_v1_152/predictions/Softmax --c_input_nodes input --c_output_nodes resnet_v1_152/logits/BiasAdd --input_shapes ?,224,224,3 --pre_process resnet_v1_152

  ## squeezenet
  python run.py --quantize --model models/squeezenet.pb --output_dir work --input_nodes data --output_nodes Prediction/softmax/Softmax --input_shapes 1,227,227,3 --pre_process squeezenet
  python run.py --validate --model models/squeezenet.pb --output_dir work --input_nodes data --output_nodes Prediction/softmax/Softmax --c_input_nodes data --c_output_nodes avg_pool/AvgPool --input_shapes ?,227,227,3 --pre_process squeezenet

  ## inception_v4
  python run.py --quantize --model models/inception_v4.pb --output_dir work --input_nodes input --output_nodes InceptionV4/Logits/Predictions --input_shapes 1,299,299,3 --pre_process inception_v4
  python run.py --validate --model models/inception_v4.pb --output_dir work --input_nodes input --output_nodes InceptionV4/Logits/Predictions --c_input_nodes input --c_output_nodes InceptionV4/Logits/Predictions --input_shapes ?,299,299,3 --label_offset 1 --pre_process inception_v4

  ## vgg_16
  python run.py --quantize --model models/vgg_16.pb --output_dir work --input_nodes input --output_nodes vgg_16/fc8/squeezed --input_shapes 1,224,224,3 --pre_process vgg
  python run.py --validate --model models/vgg_16.pb --output_dir work --input_nodes input --output_nodes vgg_16/fc8/squeezed --c_input_nodes input --c_output_nodes vgg_16/fc8/squeezed --input_shapes ?,224,224,3 --pre_process vgg

  ## vgg_19
  python run.py --quantize --model models/vgg_19.pb --output_dir work --input_nodes input --output_nodes vgg_19/fc8/squeezed --input_shapes 1,224,224,3 --pre_process vgg
  python run.py --validate --model models/vgg_19.pb --output_dir work --input_nodes input --output_nodes vgg_19/fc8/squeezed --c_input_nodes input --c_output_nodes vgg_19/fc8/squeezed --input_shapes ?,224,224,3 --pre_process vgg
  ```
