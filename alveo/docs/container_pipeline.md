## Running a pipeline of docker containers to run the various steps to prepare and deploy a caffe model, and serve with an inference server - This will not work with Vitis-AI 1.0, please use MLsuite 1.5 for this example.

1. Create a shared directory to map with the container for sharing intermediate results between various stages

  ```
  mkdir $HOME/share
  sudo chmod -R 777 $HOME/share
  ```

2. Clone ML Suite
   
  ```
  git clone https://github.com/Xilinx/ml-suite.git
  ```

3. Download a minimal validation set for [Imagenet2012](http://www.image-net.org/challenges/LSVRC/2012) using [Collective Knowledge (CK)](https://github.com/ctuning).
The same dataset is used for mlperf inference benchmarks that are using imagenet.

  You can install CK from PyPi via pip install ck or pip install ck --user.

  You can then test that it works from the command line via ck version or from your python environment as follows:

  ```
  $ python

  > import ck.kernel as ck
  > ck.version({})
  ```
  
  Refer to https://github.com/ctuning/ck for install instructions

  ```
  python -m ck pull repo:ck-env 
  python -m ck install package:imagenet-2012-val-min
  python -m ck install package:imagenet-2012-aux
  head -n 500 $HOME/CK-TOOLS/dataset-imagenet-ilsvrc2012-aux/val.txt > $HOME/CK-TOOLS/dataset-imagenet-ilsvrc2012-val-min/val_map.txt
  ```

  Resize all the images to a common dimension for Caffe
  ```
  python -m pip --no-cache-dir install opencv-python --user 
  python MLsuite/examples/caffe/resize.py $HOME/CK-TOOLS/dataset-imagenet-ilsvrc2012-val-min 256 256
  ``` 

  Move $HOME/CK-TOOLS to $HOME/share/CK-TOOLS
  ```
  mv $HOME/CK-TOOLS $HOME/share/CK-TOOLS
  ```

  Download sample models from xilinx. You may need to modify this step per your needs. 
  
  ```
  mkdir $HOME/share/models
  mkdir $HOME/share/models/caffe
  ```

  - update `modelsDir = "/opt/models/caffe/"` in MLsuite/examples/caffe/getModels.py to point to $HOME/share/models/caffe
 
  This will download sample models from xilinx to the $HOME/share/models/caffe directory. Later when invoking the docker container, we will mount one of the models we will be using, to the container.
  
3. Pull the MLsuite docker container
  
  ```
  docker pull <path-to-docker-image-on-dockerhub>
  docker load <path-to-docker-image>
  ```

In the next few steps, we will launch individual instances of the container to run the quantizer, compiler, sub-graph cutter and inference steps of the image classification. This  will allow us to horizontally as well as vertically scale the relevant steps.

While invoking docker, make sure to mount $HOME/share and $HOME/share/CK-TOOLS directories to the appropriate locations within the container as shown in the various steps below.
   
4. **Prepare for inference**

  This performs quantization, compilation and subgraph cutting in a single step.

  Quantize the model - The quantizer will generate scaling parameters for quantizing floats INT8. This is required, because FPGAs will take advantage of Fixed Point Precision, to achieve more parallelization at lower power

  Compile the Model - In this step, the network Graph (prototxt) and the Weights (caffemodel) are compiled, the compiler

  Subgraph Cutting - In this step, the original graph is cut, and a custom FPGA accelerated python layer is inserted to be used for Inference.

  ```
   docker run \
  --rm \
  --net=host \
  --privileged=true \
  -a stdin -a stdout -a stderr \
  -t \
  -v /dev:/dev \
  -v /opt/xilinx:/opt/xilinx \
  -w /opt/ml-suite \
  -v $HOME/share:/opt/share \
  -v $HOME/share/CK-TOOLS:/home/mluser/CK-TOOLS \
  -v $HOME/share/models/caffe/bvlc_googlenet:/opt/models/caffe/bvlc_googlenet \
  xilinxatg/ml_suite:ubuntu-16.04-caffe-mls-1.4 \
  bash -c 'source /opt/ml-suite/overlaybins/setup.sh && cd /opt/share/ && python run.py --prototxt /opt/models/caffe/bvlc_googlenet/bvlc_googlenet_train_val.prototxt --caffemodel /opt/models/caffe/bvlc_googlenet/bvlc_googlenet.caffemodel --prepare
  ```

  You can skip to step-8 if you want to see how to set up an inference server serving a REST API. The next few steps 5, 6, 7 walk through how to invoke the individual steps namely quantize, compile and sub-graph cutting, that we already did as a single wrapper command in step 4.

5. **Quantize the model** - The quantizer will generate scaling parameters for quantizing floats INT8. This is required, because FPGAs will take advantage of Fixed Point Precision, to achieve more parallelization at lower power

  ```
  docker run \
  --rm \
  --net=host \
  --privileged=true \
  -a stdin -a stdout -a stderr \
  -t \
  -v /dev:/dev \
  -v /opt/xilinx:/opt/xilinx \
  -w /opt/ml-suite \
  -v $HOME/share:/opt/share \
  -v $HOME/share/CK-TOOLS:/home/mluser/CK-TOOLS \
  -v $HOME/share/models/caffe/bvlc_googlenet:/opt/models/caffe/bvlc_googlenet \
  xilinxatg/ml_suite:ubuntu-16.04-caffe-mls-1.4 \
  bash -c 'source /opt/ml-suite/overlaybins/setup.sh && cd /opt/share/ && export DECENT_DEBUG=1 && vai_q_caffe quantize -model /opt/models/caffe/bvlc_googlenet/bvlc_googlenet_train_val.prototxt -weights /opt/models/caffe/bvlc_googlenet/bvlc_googlenet.caffemodel -auto_test -test_iter 1 --calib_iter 1'
  ```

  This outputs quantize_info.txt, deploy.prototxt, deploy.caffemodel to $HOME/share/quantize_results/ directory

6. **Compile the Model** - In this step, the network Graph (prototxt) and the Weights (caffemodel) are compiled, the compiler
  
  ```
  docker run \
  --rm \
  --net=host \
  --privileged=true \
  -a stdin -a stdout -a stderr \
  -t \
  -v /dev:/dev \
  -v /opt/xilinx:/opt/xilinx \
  -w /opt/ml-suite \
  -v $HOME/share:/opt/share \
  -v $HOME/CK-TOOLS:/home/mluser/CK-TOOLS \
  -v $HOME/share/models/caffe/bvlc_googlenet:/opt/models/caffe/bvlc_googlenet \
  xilinxatg/ml_suite:ubuntu-16.04-caffe-mls-1.4 \
  bash -c 'source /opt/ml-suite/overlaybins/setup.sh && cd /opt/share/ && python $VAI_ALVEO_ROOT/vai/dpuv1/tools/compile/bin/xfdnn_compiler_caffe.pyc -b 1 -i 96 -m 9 -d 256 -mix --pipelineconvmaxpool --usedeephi --quant_cfgfile quantize_results/quantize_info.txt -n quantize_results/deploy.prototxt -w quantize_results/deploy.caffemodel -g work/compiler -qz work/quantizer -C'
  ```  

7. **Subgraph Cutting** - In this step, the original graph is cut, and a custom FPGA accelerated python layer is inserted to be used for Inference.

  ```
  docker run \
  --rm \
  --net=host \
  --privileged=true \
  -a stdin -a stdout -a stderr \
  -t \
  -v /dev:/dev \
  -v /opt/xilinx:/opt/xilinx \
  -w /opt/ml-suite \
  -v $HOME/share:/opt/share \
  -v $HOME/CK-TOOLS:/home/mluser/CK-TOOLS \
  -v $HOME/share/models/caffe/bvlc_googlenet:/opt/models/caffe/bvlc_googlenet \
  xilinxatg/ml_suite:ubuntu-16.04-caffe-mls-1.4 \
  bash -c 'source /opt/ml-suite/overlaybins/setup.sh && cd /opt/share/ && python $VAI_ALVEO_ROOT/vai/dpuv1/rt/scripts/framework/caffe/xfdnn_subgraph.py --inproto quantize_results/deploy.prototxt --trainproto /opt/models/caffe/bvlc_googlenet/bvlc_googlenet_train_val.prototxt --outproto xfdnn_auto_cut_deploy.prototxt --cutAfter data --xclbin $VAI_ALVEO_ROOT/overlaybins/$MLSUITE_PLATFORM/overlay_4.xclbin --netcfg work/compiler.json --quantizecfg work/quantizer.json --weights work/deploy.caffemodel_data.h5 --profile True'
  ```      
   
8. **Inference REST API Server for Classification** - In this step, a python flask inference server is started, which takes in the caffe model as well as the prototxt from the previous step to run on the FPGA, and exposes a REST API endpoint to perform inference on input images

   This starts an inference server which can be accessed at port 5000 from end point /predict. For example: http://127.0.0.1:5000/predict

  ```
  docker run \
  --rm \
  --net=host \
  --privileged=true \
  -d \
  -v /dev:/dev \
  -v /opt/xilinx:/opt/xilinx \
  -w /opt/ml-suite \
  -v $HOME/share:/opt/share \
  -v $HOME/CK-TOOLS:/home/mluser/CK-TOOLS \
  -v $HOME/share/models/caffe/bvlc_googlenet:/opt/models/caffe/bvlc_googlenet \
  xilinxatg/ml_suite:ubuntu-16.04-caffe-mls-1.4 \
  bash -c 'source /opt/ml-suite/overlaybins/setup.sh && cd /opt/share/ && python /opt/ml-suite/examples/caffe/app.py --caffemodel /opt/models/caffe/bvlc_googlenet/bvlc_googlenet.caffemodel --prototxt xfdnn_auto_cut_deploy.prototxt --synset_words /home/mluser/CK-TOOLS/dataset-imagenet-ilsvrc2012-aux/synset_words.txt --port 5000'
  ```
   
9. **Classification (Passing an Image for Inference)** - CURL command to test the inference server REST API end point by passing images to perform inference

   ```
   curl -X POST -F image=@$HOME/share/CK-TOOLS/dataset-imagenet-ilsvrc2012-val-min/ILSVRC2012_val_00000001.JPEG 'http://localhost:5000/predict'
   ```
