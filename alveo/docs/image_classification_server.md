## Image Classification with Xilinx FPGA  

The Xilinx Machine Learning (ML) Suite provides users with the tools to develop and deploy Machine Learning applications for Real-time Inference. It provides support for many common machine learning frameworks such as Caffe, Tensorflow, and MXNet.

In this tutorial, we will go through using the Xilinx MLSuite docker container tools on AWS, to quantize and compile any existing caffe model, create an inference server and perform inference.  If you are running on-premise, you will replace steps 1 & 2 with appropriate steps to SSH into your on-prem Xilinx machine. For on-premise instructions, refer to docs/xrt.md under the MLSuite directory within the Xilinx MLSuite container.


### Getting Started - Launch the AMI

Subscribe to and launch the Xilinx MLSuite AMI on any of the AWS Xilinx F1 Instances, from here:

https://aws.amazon.com/marketplace/pp/B077FM2JNS

### EC2 Instance Access

To get started, launch an AWS F1 instance using this AMI from the EC2 console. If you are not familiar with this process please review the AWS documentation provided here:

http://docs.aws.amazon.com/AWSEC2/latest/UserGuide/launching-instance.html

Accessing the instance via SSH:

```
ssh -i <path to your pem file> centos@{EC2 Instance Public IP}
```
 
### Setup the MLSuite Docker Container

The ML Suite Docker container provides users with a largely self contained software environment for running inference on Xilinx FPGAs.
The only external dependencies are:
- docker-ce
- [Xilinx XRT 2018.2 (Xilinx Run Time)](xrt.md)

1. Install Docker

   - [ubuntu](https://docs.docker.com/install/linux/docker-ce/ubuntu/#install-docker-ce)
   - [centos](https://docs.docker.com/install/linux/docker-ce/centos/#install-docker-ce)

   Note: Ensure /var/lib/docker has sufficient space (Should be > 5GB), if not move your Docker Root elsewhere.

2. Download the appropriate ML Suite Container from xilinx.com

   - [ML Suite Caffe Container](https://www.xilinx.com/member/forms/download/eula-xef.html?filename=xilinx-ml-suite-ubuntu-16.04-xrt-2018.2-caffe-mls-1.4.tar.gz)

3. Load the container
   ```
   # May need sudo
   $ docker load < xilinx-ml-suite-ubuntu-16.04-xrt-2018.2-caffe-mls-1.4.tar.gz
   ```

4. Use the provided [script](../docker_run.sh) to launch and interact with the container
   ```
   # May need sudo
   $ ./docker_run.sh

   # The script by default will launch a container with the --rm flag, meaning after exiting, the container will be removed.
   # Changes to source files will be lost.
   # A directory $VAI_ALVEO_ROOT/share will be mounted in the container and can be used for easy file transfer between container and host.
   ```

5. Download a minimal validation set for [Imagenet2012](http://www.image-net.org/challenges/LSVRC/2012) using [Collective Knowledge (CK)](https://github.com/ctuning).
The same dataset is used for mlperf inference benchmarks that are using imagenet.

```
python -m ck pull repo:ck-env
python -m ck install package:imagenet-2012-val-min
python -m ck install package:imagenet-2012-aux
head -n 500 $HOME/CK-TOOLS/dataset-imagenet-ilsvrc2012-aux/val.txt > $HOME/CK-TOOLS/dataset-imagenet-ilsvrc2012-val-min/val_map.txt

# Resize all the images to a common dimension for Caffe

python $VAI_ALVEO_ROOT/examples/caffe/resize.py $HOME/CK-TOOLS/dataset-imagenet-ilsvrc2012-val-min 256 256
```

6. Get Samples Models 

```
python $VAI_ALVEO_ROOT/examples/caffe/getModels.py
```

7. Setup ml-suite Environment Variables

```
source $VAI_ALVEO_ROOT/overlaybins/setup.sh
```

### Prepare for inference

The following example uses the googlenet example. You can try the flow with also the other models found in /opt/models/caffe/ directory.

At any given point, you can always refer to the README.md in the top-level directory of MLSuite as well as the ones in the various sub-folders, for detailed steps and options, etc. 

This performs quantization, compilation and subgraph cutting in a single step. To run a Caffe model on the FPGA, it needs to be quantized, compiled, and a new graph needs to be generated. The new graph is similar to the original, with the FPGA subgraph removed, and replaced with a custom Python layer.

  Quantize the model - The quantizer will generate scaling parameters for quantizing floats INT8. This is required, because FPGAs will take advantage of Fixed Point Precision, to achieve more parallelization at lower power

  Compile the Model - In this step, the network Graph (prototxt) and the Weights (caffemodel) are compiled, the compiler

  Subgraph Cutting - In this step, the original graph is cut, and a custom FPGA accelerated python layer is inserted to be used for Inference.

  ```
  cd $VAI_ALVEO_ROOT/examples/caffe 
  python run.py --prototxt /opt/models/caffe/bvlc_googlenet/bvlc_googlenet_train_val.prototxt --caffemodel /opt/models/caffe/bvlc_googlenet/bvlc_googlenet.caffemodel --prepare
  ```

### Running an Inference Server for Classification

In this step, a python flask inference server is started, and the caffe model as well as the prototxt from the previous step are run on the FPGA to perform inference on an input image.

This starts an inference server which can be accessed at port 5000 from end point /predict. For example: http://127.0.0.1:5000/predict

   ```
   cd $VAI_ALVEO_ROOT/examples/caffe/REST
   ln -s ../work .
   python app.py --caffemodel /opt/models/caffe/bvlc_googlenet/bvlc_googlenet.caffemodel --prototxt $VAI_ALVEO_ROOT/examples/caffe/work/xfdnn_auto_cut_deploy.prototxt --synset_words /home/mluser/CK-TOOLS/dataset-imagenet-ilsvrc2012-aux/synset_words.txt --port 5000
   ```

You can switch the above command to running in the background, and you can use a CURL command perform inference on an input image, passed to the REST inference server. Note that you wont just type the inference server URL on the browser, but only use the command line options shown below to pass an input image to classify. Feel free to take this as a starting point and modify as needed for your inference server needs. 

### Classification (Sending sample image requests to the Inference Server)

   ```
   curl -X POST -F image=@$HOME/CK-TOOLS/dataset-imagenet-ilsvrc2012-val-min/ILSVRC2012_val_00000001.JPEG 'http://localhost:5000/predict'
   ```

You can switch the above command to running in the background, and you can use a CURL command perform inference on an input image, passed to the REST inference server. Note that you wont just type the inference server URL on the b
rowser, but only use the command line options shown below to pass an input image to classify. Feel free to take thi
s as a starting point and modify as needed for your inference server needs.

### Benchmarking

   Run the benchmark.py script in /opt/ml-suite/examples/caffe/ directory, which will send a sample input image to the inference server repeatedly while measuring response times, and finally calculating the average response time.

  ```
  cd $VAI_ALVEO_ROOT/examples/caffe/REST
  python benchmark.py --rest_api_url http://localhost:5000/predict --image_path /opt/share/CK-TOOLS/dataset-imagenet-ilsvrc2012-val-min/ILSVRC2012_val_00000001.JPEG
  ```

