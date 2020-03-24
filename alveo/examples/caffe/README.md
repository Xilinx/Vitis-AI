## Getting Started to prepare and deploy a trained Caffe model for FPGA acceleration

### Running Caffe Benchmark Models
This directory provides scripts for running several well known models on the FPGA.
For published results, and details on running a full accuracy evaluation, please see these [instructions](Benchmark_README.md).

1. **One time setup**

   Download a minimal validation set for [Imagenet2012](http://www.image-net.org/challenges/LSVRC/2012) using [Collective Knowledge (CK)](https://github.com/ctuning).

   ```
   cd  $VAI_ALVEO_ROOT/examples/caffe
   python -m ck pull repo:ck-env
   python -m ck install package:imagenet-2012-val-min
   python -m ck install package:imagenet-2012-aux
   head -n 500 $HOME/CK-TOOLS/dataset-imagenet-ilsvrc2012-aux/val.txt > $HOME/CK-TOOLS/dataset-imagenet-ilsvrc2012-val-min/val_map.txt
   ```

   Resize all the images to a common dimension

   ```
   python resize.py $HOME/CK-TOOLS/dataset-imagenet-ilsvrc2012-val-min 256 256
   ```

   Get Samples Models

   ```
   python getModels.py
   python replace_mluser.py --modelsdir models
   ```

   Setup the Environment

   ```
   source $VAI_ALVEO_ROOT/overlaybins/setup.sh
   ```

After the setup, run through a sample end to end caffe classification example using the following steps that demonstrates preparing and deploying a trained Caffe model for FPGA acceleration using Xilinx MLSuite**

  The following example uses the googlenet example. You can try the flow with also the other models found in /opt/models/caffe/ directory.

  ```
  cd $VAI_ALVEO_ROOT/examples/caffe 
  ``` 

2. **Prepare for inference**

  This performs quantization, compilation and subgraph cutting in a single step. To run a Caffe model on the FPGA, it needs to be quantized, compiled, and a new graph needs to be generated. The new graph is similar to the original, with the FPGA subgraph removed, and replaced with a custom Python layer.
  
  Quantize the model - The quantizer will generate scaling parameters for quantizing floats INT8. This is required, because FPGAs will take advantage of Fixed Point Precision, to achieve more parallelization at lower power

  Compile the Model - In this step, the network Graph (prototxt) and the Weights (caffemodel) are compiled, the compiler
 
  Subgraph Cutting - In this step, the original graph is cut, and a custom FPGA accelerated python layer is inserted to be used for Inference.
  
  The above three steps are invoked by using the --prepare switch  
  
  If you plan to work on several models, you can use the `--output_dir` switch to generate and store model artifacts in seperate directories. By default, output_dir is ./work

  ```
   python run.py --prototxt models/bvlc_googlenet/bvlc_googlenet_train_val.prototxt --caffemodel models/bvlc_googlenet/bvlc_googlenet.caffemodel --prepare --output_dir work
  ```

3. **Validate accuracy** - Run the validation set for 10 iterations on the FPGA

  ```
   python run.py --validate --output_dir work --numBatches 10
  ```

4. **Inference** - Run a single image on the FPGA

  ```
  python run.py --image ../deployment_modes/dog.jpg
  ```


 Note: The above instruction assumes that the --output_dir switch was not used, and artifacts were generated in ./work
