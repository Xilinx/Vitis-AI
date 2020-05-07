## Getting Started to prepare and deploy a trained tensorflow model for FPGA acceleration

### Running tensorflow Benchmark Models
This directory provides scripts for running several well known models on the FPGA.
For published results, and details on running a full accuracy evaluation, please see these [instructions](Benchmark_README.md).

1. **One time setup**

   Download a minimal validation set for [Imagenet2012](http://www.image-net.org/challenges/LSVRC/2012) using [Collective Knowledge (CK)](https://github.com/ctuning).

   ```
   $ python -m ck pull repo:ck-env
   $ python -m ck install package:imagenet-2012-val-min
   $ python -m ck install package:imagenet-2012-aux
   $ head -n 500 ~/CK-TOOLS/dataset-imagenet-ilsvrc2012-aux/val.txt > ~/CK-TOOLS/dataset-imagenet-ilsvrc2012-val-min/val.txt
   ```

   Get Samples Models. Models will be downloaded into models/ directory.

   ```
   $ python getModels.py
   ```

   Setup the Environment

   ```
   $ source $VAI_ALVEO_ROOT/overlaybins/setup.sh
   ```

After the setup, run through a sample end to end tensorflow classification example using the following steps that demonstrates preparing and deploying a trained tensorflow model for FPGA acceleration using Xilinx MLSuite**

  The following example uses the googlenet example. You can try the flow with also the other models found in /opt/models/tensorflow/ directory.

  ```
  $ cd $VAI_ALVEO_ROOT/examples/tensorflow
  ```
2. **Inspect TensorFlow model**

   Inspect the tensorflow model to get input and output node(s), and input nodes shape. 
   ```
   $ ./inspect_tf_model.sh <tensorflow_model>
   ```
   
   ```
   $ ./inspect_tf_model.sh models/inception_v1_baseline.pb
   ```
   
3. **Quantize for inference**

  To run a tensorflow model on the FPGA, it needs to be quantized.

  Quantize the model - The quantizer will generate scaling parameters for quantizing floats INT8. This is required, because FPGAs will take advantage of Fixed Point Precision, to achieve more parallelization at lower power

  The above step is invoked by using the --quantize switch. If you plan to work on several models, you can use the `--output_dir` switch to generate and store model artifacts in seperate directories. By default, output_dir is ./work

  ```
  $ python run.py --quantize --model models/inception_v1_baseline.pb --pre_process inception_v1 --output_dir work --input_nodes data --output_nodes loss3_loss3 --input_shapes 1,224,224,3 --batch_size 16
  ```

4. **Partition, Compile, and Run Inference**
  This performs compilation and subgraph partitioning in a single step. To run a tensorflow model on the FPGA, it needs to be compiled and a new graph needs to be generated. The new graph is similar to the original, with the FPGA subgraph removed, and replaced with a custom Python layer.

  Compile the Model - In this step, the network Graph (frozen pb file) are compiled.

  Subgraph Partitioning - In this step, the original graph is cut, and a custom FPGA accelerated python layer is inserted to be used for Inference.

  The above two steps are invoked by using the --partition switch. If you plan to work on several models, you can use the `--output_dir` switch to generate and store model artifacts in seperate directories. By default, output_dir is ./work

  ```
  $ python run.py --validate --model models/inception_v1_baseline.pb --pre_process inception_v1 --output_dir work --input_nodes data --output_nodes loss3_loss3 --c_input_nodes data --c_output_nodes pool5_7x7_s1 --input_shapes ?,224,224,3

  # Note: You might need to change run.py , and adjust the key name of the returned dictionary in the calib_input, preprocess_default and preprocess_inception functions. Adjust the key name to match the name of the input tensor to your graph.

  ```


 5. **Other Models** - scripts for running compilation, partitioning and inference with other models provided by getModels.py

  ```
  # resnet_50
  $ python run.py --quantize --model models/resnet50_baseline.pb --output_dir work --input_nodes data --output_nodes prob --input_shapes 1,224,224,3 --pre_process resnet50
  $ python run.py --validate --model models/resnet50_baseline.pb --output_dir work --input_nodes data --output_nodes prob --c_input_nodes data --c_output_nodes prob --input_shapes ?,224,224,3 --pre_process resnet50

  # resnet_101
  $ python run.py --quantize --model models/resnet_v1_101.pb --output_dir work --input_nodes input --output_nodes resnet_v1_101/predictions/Softmax --input_shapes 1,224,224,3 --pre_process resnet_v1_101
  $ python run.py --validate --model models/resnet_v1_101.pb --output_dir work --input_nodes input --output_nodes resnet_v1_101/predictions/Softmax --c_input_nodes input --c_output_nodes resnet_v1_101/logits/BiasAdd --input_shapes ?,224,224,3 --pre_process resnet_v1_101

  # resnet_152
  $ python run.py --quantize --model models/resnet_v1_152.pb --output_dir work --input_nodes input --output_nodes resnet_v1_152/predictions/Softmax --input_shapes 1,224,224,3 --pre_process resnet_v1_152
  $ python run.py --validate --model models/resnet_v1_152.pb --output_dir work --input_nodes input --output_nodes resnet_v1_152/predictions/Softmax --c_input_nodes input --c_output_nodes resnet_v1_152/logits/BiasAdd --input_shapes ?,224,224,3 --pre_process resnet_v1_152

  # squeezenet
  $ python run.py --quantize --model models/squeezenet.pb --output_dir work --input_nodes data --output_nodes Prediction/softmax/Softmax --input_shapes 1,227,227,3 --pre_process squeezenet
  $ python run.py --validate --model models/squeezenet.pb --output_dir work --input_nodes data --output_nodes Prediction/softmax/Softmax --c_input_nodes data --c_output_nodes avg_pool/AvgPool --input_shapes ?,227,227,3 --pre_process squeezenet

  # inception_v4
  $ python run.py --quantize --model models/inception_v4.pb --output_dir work --input_nodes input --output_nodes InceptionV4/Logits/Predictions --input_shapes 1,299,299,3 --pre_process inception_v4
  $ python run.py --validate --model models/inception_v4.pb --output_dir work --input_nodes input --output_nodes InceptionV4/Logits/Predictions --c_input_nodes input --c_output_nodes InceptionV4/Logits/Predictions --input_shapes ?,299,299,3 --label_offset 1 --pre_process inception_v4
  ```
