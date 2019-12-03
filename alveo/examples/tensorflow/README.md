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

2. **Quantize for inference**

  To run a tensorflow model on the FPGA, it needs to be quantized.

  Quantize the model - The quantizer will generate scaling parameters for quantizing floats INT8. This is required, because FPGAs will take advantage of Fixed Point Precision, to achieve more parallelization at lower power

  The above step is invoked by using the --quantize switch. If you plan to work on several models, you can use the `--output_dir` switch to generate and store model artifacts in seperate directories. By default, output_dir is ./work

  ```
  $ python run.py --quantize --model models/inception_v1_baseline.pb --pre_process inception_v1 --output_dir work --input_nodes data --output_nodes loss3_loss3 --input_shapes 1,224,224,3 --batch_size 16
  ```

3. **Partition, Compile, and Run Inference**
  This performs compilation and subgraph partitioning in a single step. To run a tensorflow model on the FPGA, it needs to be compiled and a new graph needs to be generated. The new graph is similar to the original, with the FPGA subgraph removed, and replaced with a custom Python layer.

  Compile the Model - In this step, the network Graph (frozen pb file) are compiled.

  Subgraph Partitioning - In this step, the original graph is cut, and a custom FPGA accelerated python layer is inserted to be used for Inference.

  The above two steps are invoked by using the --partition switch. If you plan to work on several models, you can use the `--output_dir` switch to generate and store model artifacts in seperate directories. By default, output_dir is ./work

  ```
  $ python run.py --validate --model models/inception_v1_baseline.pb --pre_process inception_v1 --output_dir work --input_nodes data --output_nodes loss3_loss3 --c_input_nodes data --c_output_nodes pool5_7x7_s1 --input_shapes ?,224,224,3

  # Note: You might need to change run.py , and adjust the key name of the returned dictionary in the calib_input, preprocess_default and preprocess_inception functions. Adjust the key name to match the name of the input tensor to your graph.

  ```



4. **Benchmark FPGA performance** - evaluate network throughput and/or latency in a streaming deployment scenario (FPGA only)

  ```
  cd $VAI_ALVEO_ROOT/examples/deployment_modes && \
  ./run.sh -ns 1 \
    -t streaming_classify_fpgaonly -v -x \
    -d $HOME/CK-TOOLS/dataset-imagenet-ilsvrc2012-val-min \
    -cw $VAI_ALVEO_ROOT/examples/caffe/work/deploy.caffemodel_data.h5 \
    -cn $VAI_ALVEO_ROOT/examples/caffe/work/compiler.json \
    -cq $VAI_ALVEO_ROOT/examples/caffe/work/quantizer.json \
    | python $VAI_ALVEO_ROOT/vai/dpuv1/rt/scripts/speedometer.py
  ```

 `-ns 1` tells the script to use 1 stream, which sends 1 image to each PE. This will achieve minimum latency.
 To maximize throughput, try increasing the number of streams with `-ns` (e.g., `-ns 4`) until `FPGA utilization` reaches 100%. After `FPGA utilization` reaches 100%, increasing the number of streams no longer improves throughput and only hurts latency.

 To exit the streaming demo, press `CTRL-Z` and type `kill -9 %%`.

 6. **Benchmark end-to-end performance** - evaluate end-to-end network throughput and/or latency in a streaming deployment scenario.

 This demo includes pre-processing and post-processing, which run on CPU. In some cases, CPU code may need additional optimizations in order to maximize the utilization of FPGA.

  ```
  cd $VAI_ALVEO_ROOT/examples/deployment_modes && \
  ./run.sh -ns 1 \
    -t streaming_classify -v -x \
    -d $HOME/CK-TOOLS/dataset-imagenet-ilsvrc2012-val-min \
    -cw $VAI_ALVEO_ROOT/examples/caffe/work/deploy.caffemodel_data.h5 \
    -cn $VAI_ALVEO_ROOT/examples/caffe/work/compiler.json \
    -cq $VAI_ALVEO_ROOT/examples/caffe/work/quantizer.json \
    | python $VAI_ALVEO_ROOT/vai/dpuv1/rt/scripts/speedometer.py
  ```

 To exit the streaming demo, press `CTRL-Z` and type `kill -9 %%`.


 Note: The above instruction assumes that the --output_dir switch was not used, and artifacts were generated in ./work
