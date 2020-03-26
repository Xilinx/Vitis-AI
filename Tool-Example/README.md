# Vitis AI Tool Example

## 1. Prerequisite
- Docker service is available on host machine. For detailed steps, please refer to [Install Docker](/doc/install_docker/README.md) page
- Vitis AI CPU/GPU docker image is correctly loaded and launched. For detailed steps, please refer to [Load and Run Docker](/doc/install_docker/load_run_docker.md)

## 2. File Preparation

Caffe and TensorFlow Resnet50 are used in following examples targeting both edge DPUv2 and Alveo DPUv3. Network packages can be found on [Xilinx AI Model Zoo](https://github.com/Xilinx/AI-Model-Zoo). The script is provided to download them. 

Reference files used in the examples can be found in [Vitis AI Tool Example](https://github.com/Xilinx/Vitis-AI/tree/master/Tool-Example). 

```
Tool-Example/
├── 0_download_model.sh
├── 1_caffe_quantize_for_v2.sh
├── 1_caffe_quantize_for_v3.sh
├── 2_caffe_compile_for_v2.sh
├── 2_caffe_compile_for_v3.sh
├── 3_tf_eval_frozen_graph.sh
├── 4_tf_quantize.sh
├── 5_tf_eval_quantize_graph.sh
├── 6_tf_compile_for_v2.sh
├── 6_tf_compile_for_v3.sh
├── example_file
│   ├── input_fn.py
│   ├── resnet_eval.py
│   └── trainval.prototxt
└── images
    ├── caffe_calib.txt
    ├── list.txt
    └── tf_calib.txt

```

In order to conduct calibration and evaluation in quantization phase, please download images and labels from ImageNet and place them under folder `Tool-Example/images/`. *ILSVRC2012_val_00000001.JPEG* to *ILSVRC2012_val_00001000.JPEG* are used in following examples. 

After docker is started, please use **0_download_model.sh** to download two models and set several necessary parameters, such as path to the networks. 

`. 0_download_model.sh`

```
cf_resnet50_imagenet_224_224_7.7G
├── code
│   ├── gen_data
│   │   ├── gen_data.py
│   │   ├── get_dataset.sh
│   │   └── imagenet_class_index.json
│   ├── test
│   │   └── test.sh
│   └── train
│       ├── solver.prototxt
│       └── train.sh
├── data
├── float
│   ├── test.prototxt
│   ├── trainval.caffemodel
│   ├── trainval.prototxt
│   └── trainval.prototxt.bak
├── quantized
│   ├── deploy.caffemodel
│   ├── deploy.prototxt
│   ├── quantize_test.prototxt
│   ├── quantize_train_test.caffemodel
│   └── quantize_train_test.prototxt
└── readme.md

tf_resnetv1_50_imagenet_224_224_6.97G
├── code
│   ├── conda_config
│   │   └── enveriment.yaml
│   ├── gen_data
│   │   ├── gen_data.py
│   │   ├── get_dataset.sh
│   │   └── imagenet_class_index.json
│   └── test
│       ├── eval_tf_classification_models_alone.py
│       └── run_eval_pb.sh
├── data
├── float
│   └── resnet_v1_50_inference.pb
├── quantized
│   ├── deploy_model.pb
│   └── quantize_eval_model.pb
└── readme.md
```

## 3. Caffe Workflow

In order to use Caffe tools, please use following command to activate caffe tool conda environment

`conda activate vitis-ai-caffe`


- ### Quantization for DPU edge overlay (DPUv2) 

Run script **1_caffe_quantize_for_v2.sh** to calibrate, quantize and evaluate Resnet50 for DPUv2. 

`sh 1_caffe_quantize_for_v2.sh`

Detailed commands used in the script are as below:

```
#For DPUv2
vai_q_caffe quantize -model ${CF_NETWORK_PATH}/float/trainval.prototxt \
                     -weights ${CF_NETWORK_PATH}/float/trainval.caffemodel \
                     -output_dir ${CF_NETWORK_PATH}/vai_q_output_dpuv2 \
                     -calib_iter 100 \
                     -test_iter 100 \
                     -auto_test \
```

Main outputs are as below:

```
I1128 19:03:59.594540  1634 net.cpp:98] Initializing net from parameters:
.....................
I1128 19:04:00.902045  1634 decent_q.cpp:213] Start Calibration
I1128 19:04:01.076759  1634 decent_q.cpp:237] Calibration iter: 1/100 ,loss: 1.22658
I1128 19:04:01.174342  1634 decent_q.cpp:237] Calibration iter: 2/100 ,loss: 0.800063
I1128 19:04:01.259806  1634 decent_q.cpp:237] Calibration iter: 3/100 ,loss: 0.951107
.....................
I1128 19:04:11.308645  1634 net_test.cpp:411] Test Start, total iterations: 100
I1128 19:04:11.308647  1634 net_test.cpp:352] Testing ...
I1128 19:04:11.394686  1634 net_test.cpp:373] Test iter: 1/100, top-1 = 0.7
I1128 19:04:11.394709  1634 net_test.cpp:373] Test iter: 1/100, top-5 = 0.9
.....................
I1128 19:04:16.615674  1634 net_test.cpp:381] Test Results:
I1128 19:04:16.615676  1634 net_test.cpp:382] Loss: 0
I1128 19:04:16.615679  1634 net_test.cpp:396] top-1 = 0.744
I1128 19:04:16.615682  1634 net_test.cpp:396] top-5 = 0.927
I1128 19:04:16.615684  1634 net_test.cpp:419] Test Done!
I1128 19:04:16.714090  1634 decent_q.cpp:391] Start Deploy
I1128 19:04:17.001940  1634 decent_q.cpp:399] Deploy Done!
--------------------------------------------------
Output Quantized Train&Test Model:   "cf_resnet50_imagenet_224_224_7.7G/vai_q_output_dpuv2/quantize_train_test.prototxt"
Output Quantized Train&Test Weights: "cf_resnet50_imagenet_224_224_7.7G/vai_q_output_dpuv2/quantize_train_test.caffemodel"
Output Deploy Weights: "cf_resnet50_imagenet_224_224_7.7G/vai_q_output_dpuv2/deploy.caffemodel"
Output Deploy Model:   "cf_resnet50_imagenet_224_224_7.7G/vai_q_output_dpuv2/deploy.prototxt"
```

The **quantize_train_test** model can be used for INT8 evaluation, finetune or DPU result emulation. The **deploy** model can be used to generate DPU elf file. 


- ### Compilation for DPU edge overlay (DPUv2)

Run script **2_caffe_compile_for_v2.sh** to compile Resnet50 deploy model for Vitis AI ZCU102 pre-built image as example. If other configuration is perferred, please refer to [UG1414 Vitis AI User Guide](https://www.xilinx.com/html_docs/vitis_ai/1_1/zkj1576857115470.html) about how to generate corresponding arch file and compile the model.

`sh 2_caffe_compile_for_v2.sh`

Detailed commands used in the script are as below:

```
NET_NAME=resnet50
EDGE_TARGET=ZCU102
DEPLOY_MODEL_PATH=vai_q_output_dpuv2

EDGE_ARCH=/opt/vitis_ai/compiler/arch/dpuv2/${EDGE_TARGET}/${EDGE_TARGET}.json

#DPUv2
vai_c_caffe --prototxt ${CF_NETWORK_PATH}/${DEPLOY_MODEL_PATH}/deploy.prototxt \
            --caffemodel ${CF_NETWORK_PATH}/${DEPLOY_MODEL_PATH}/deploy.caffemodel \
            --arch ${EDGE_ARCH} \
            --output_dir ${CF_NETWORK_PATH}/vai_c_output_${EDGE_TARGET}/ \
            --net_name ${NET_NAME} \
            --options "{'save_kernel':''}"
```

After compilation is finished, compiler will report kernel information as below. 

```
**************************************************
* VITIS_AI Compilation - Xilinx Inc.
**************************************************
kernel list info for network "resnet50"
                               Kernel ID : Name
                                       0 : resnet50

                             Kernel Name : resnet50
--------------------------------------------------------------------------------
                             Kernel Type : DPUKernel
                               Code Size : 0.67MB
                              Param Size : 24.35MB
                           Workload MACs : 7715.95MOPS
                         IO Memory Space : 2.25MB
                              Mean Value : 104, 117, 123,
                      Total Tensor Count : 56
                Boundary Input Tensor(s)   (H*W*C)
                               data:0(0) : 224*224*3

               Boundary Output Tensor(s)   (H*W*C)
                             fc1000:0(0) : 1*1*1000

                        Total Node Count : 55
                           Input Node(s)   (H*W*C)
                                conv1(0) : 224*224*3

                          Output Node(s)   (H*W*C)
                               fc1000(0) : 1*1*1000

```

Important items (e.g. Kernel Name, Input Node(s) and Output Node(s)) will be used in programming. Generated elf file will be used in cross-compilation.

- ### Quantization for DPU Alveo overlay (DPUv3) 

Run script **1_caffe_quantize_for_v3.sh** to calibrate, quantize and evaluate Resnet50 for DPUv3. 

`sh 1_caffe_quantize_for_v3.sh`

Detailed commands used in the script are as below:

```
#For DPUv3
vai_q_caffe quantize -model ${CF_NETWORK_PATH}/float/trainval.prototxt \
                     -weights ${CF_NETWORK_PATH}/float/trainval.caffemodel \
                     -output_dir ${CF_NETWORK_PATH}/vai_q_output_dpuv3 \
                     -calib_iter 100 \
                     -test_iter 100 \
                     -auto_test \
                     -keep_fixed_neuron
```

The '-keep_fixed_neuron' option needs to be added when targeting Alveo DPUv3. 

Main outputs are as below:

```
I1128 19:03:59.594540  1634 net.cpp:98] Initializing net from parameters:
.....................
I1128 19:04:00.902045  1634 decent_q.cpp:213] Start Calibration
I1128 19:04:01.076759  1634 decent_q.cpp:237] Calibration iter: 1/100 ,loss: 1.22658
I1128 19:04:01.174342  1634 decent_q.cpp:237] Calibration iter: 2/100 ,loss: 0.800063
I1128 19:04:01.259806  1634 decent_q.cpp:237] Calibration iter: 3/100 ,loss: 0.951107
.....................
I1128 19:04:11.308645  1634 net_test.cpp:411] Test Start, total iterations: 100
I1128 19:04:11.308647  1634 net_test.cpp:352] Testing ...
I1128 19:04:11.394686  1634 net_test.cpp:373] Test iter: 1/100, top-1 = 0.7
I1128 19:04:11.394709  1634 net_test.cpp:373] Test iter: 1/100, top-5 = 0.9
.....................
I1128 19:04:16.615674  1634 net_test.cpp:381] Test Results:
I1128 19:04:16.615676  1634 net_test.cpp:382] Loss: 0
I1128 19:04:16.615679  1634 net_test.cpp:396] top-1 = 0.744
I1128 19:04:16.615682  1634 net_test.cpp:396] top-5 = 0.927
I1128 19:04:16.615684  1634 net_test.cpp:419] Test Done!
I1128 19:04:16.714090  1634 decent_q.cpp:391] Start Deploy
I1128 19:04:17.001940  1634 decent_q.cpp:399] Deploy Done!
--------------------------------------------------
Output Quantized Train&Test Model:   "cf_resnet50_imagenet_224_224_7.7G/vai_q_output_dpuv3/quantize_train_test.prototxt"
Output Quantized Train&Test Weights: "cf_resnet50_imagenet_224_224_7.7G/vai_q_output_dpuv3/quantize_train_test.caffemodel"
Output Deploy Weights: "cf_resnet50_imagenet_224_224_7.7G/vai_q_output_dpuv3/deploy.caffemodel"
Output Deploy Model:   "cf_resnet50_imagenet_224_224_7.7G/vai_q_output_dpuv3/deploy.prototxt"
```

- ### Compilation for DPU Alveo overlay (DPUv3)

Run script **2_caffe_compile_for_v3.sh** to compile Resnet50 deploy model for Vitis AI U50 DPUv3 as example.

`sh 2_caffe_compile_for_v3.sh`

Detailed commands used in the script are as below:

```
ALVEO_TARGET=dpuv3e
NET_NAME=resnet50
DEPLOY_MODEL_PATH_v3=vai_q_output_dpuv3

ALVEO_ARCH=/opt/vitis_ai/compiler/arch/${ALVEO_TARGET}/arch.json

#DPUv3
vai_c_caffe --prototxt ${CF_NETWORK_PATH}/${DEPLOY_MODEL_PATH_v3}/deploy.prototxt \
            --caffemodel ${CF_NETWORK_PATH}/${DEPLOY_MODEL_PATH_v3}/deploy.caffemodel \
            --arch ${ALVEO_ARCH} \
            --output_dir ${CF_NETWORK_PATH}/vai_c_output_${ALVEO_TARGET} \
            --net_name ${NET_NAME} \
```

After compilation is finished, compiler will report kernel information as below. 

```
**************************************************
* VITIS_AI Compilation - Xilinx Inc.
**************************************************
[INFO] ignore pretty_errors package
[INFO] Namespace(infer_shape=False, inputs_shape=None, layout='NCHW', model_files=['cf_resnet50_imagenet_224_224_7.7G/vai_q_output_dpuv3/deploy.caffemodel'], model_type='caffe', out_filename='cf_resnet50_imagenet_224_224_7.7G/vai_c_output_dpuv3e/resnet50_org.xmodel', proto='cf_resnet50_imagenet_224_224_7.7G/vai_q_output_dpuv3/deploy.prototxt')
[INFO] caffe model: cf_resnet50_imagenet_224_224_7.7G/vai_q_output_dpuv3/deploy.caffemodel
[INFO] caffe model: cf_resnet50_imagenet_224_224_7.7G/vai_q_output_dpuv3/deploy.prototxt
[INFO] parse raw model :100%|######################################################################################################################################| 194/194 [00:04<00:00, 41.88it/s]
[INFO] generate xmodel :100%|######################################################################################################################################| 196/196 [00:00<00:00, 783.95it/s]
[INFO] generate xmodel: /workspace/Tool-Example/cf_resnet50_imagenet_224_224_7.7G/vai_c_output_dpuv3e/resnet50_org.xmodel
[UNILOG][INFO] The compiler log will be dumped at "/tmp/shuaizh/log/xcompiler-20200325-215821-279"
[UNILOG][INFO] Target architecture: DPUv3e B4096
[UNILOG][INFO] Compile mode: dpu
[UNILOG][INFO] Begin to compile...
[UNILOG][INFO] Graph name: deploy, with op num: 412
[UNILOG][INFO] Compile done.
[UNILOG][INFO] The meta json is saved to "/workspace/Tool-Example/cf_resnet50_imagenet_224_224_7.7G/vai_c_output_dpuv3e/meta.json"
[UNILOG][INFO] The compiled xmodel is saved to "/workspace/Tool-Example/cf_resnet50_imagenet_224_224_7.7G/vai_c_output_dpuv3e/resnet50.xmodel"
[UNILOG][INFO] The compiled xmodel's md5sum is 088b40b1d65047ac87061002a61605f9, and been saved to "/workspace/Tool-Example/cf_resnet50_imagenet_224_224_7.7G/vai_c_output_dpuv3e/md5sum.txt"
```


## 4. TensorFlow Workflow


Due to the nature of TensorFlow, the input function and evaluation scripts need to be prepared for networks repectively. The files used in this example are listed in **example_file**.

In order to use TensorFlow and tools, please use following command to activate TensorFlow tool conda environment

`conda activate vitis-ai-tensorflow`

- ### Float model evaluation 

Run script **3_tf_eval_frozen_graph.sh** to evaluate Resnet50 float model. 

`sh 3_tf_eval_frozen_graph.sh`

Detailed commands used in the script are as below:
```
EVAL_SCRIPT_PATH=example_file

python ${EVAL_SCRIPT_PATH}/resnet_eval.py \
       --input_frozen_graph ${TF_NETWORK_PATH}/float/resnet_v1_50_inference.pb \
       --input_node input \
       --output_node resnet_v1_50/predictions/Reshape_1 \
       --eval_batches 10 \
       --batch_size 10 \
       --eval_image_dir images/ \
       --eval_image_list images/list.txt \
```

The evluation result is as below: 

```
22:05:15.074920: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1325] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 15022 MB memory) -> physical GPU (device: 0, name: Tesla V100-PCIE-16GB, pci bus id: 0000:3b:00.0, compute capability: 7.0)
22:05:15.076890: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x55ed86064a60 initialized for platform CUDA (this does not guarantee that XLA will be used). Devices:
22:05:15.076903: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Tesla V100-PCIE-16GB, Compute Capability 7.0
N/A% (0 of 10) |                                                                                                                                                              | Elapsed Time: 0:00:00 ETA:  --:--:-- 22:05:16.625592: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcudnn.so.7
Relying on driver to perform ptx compilation. This message will be only logged once.
22:05:17.978646: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcublas.so.10.0
100% (10 of 10) |#############################################################################################################################################################| Elapsed Time: 0:00:03 Time:  0:00:03
Accuracy: 0.7
```


- ### Quantization

Run script **4_tf_quantize.sh** to calibrate and quantize Resnet50. 

`sh 4_tf_quantize.sh`

Detailed commands used in the script are as below:

```
vai_q_tensorflow quantize --input_frozen_graph ${TF_NETWORK_PATH}/float/resnet_v1_50_inference.pb \
                          --input_fn example_file.input_fn.calib_input \
                          --output_dir ${TF_NETWORK_PATH}/vai_q_output \
                          --input_nodes input \
                          --output_nodes resnet_v1_50/predictions/Reshape_1 \
                          --input_shapes ?,224,224,3 \
                          --calib_iter 100 \
```

Main outputs are as below:

```
INFO: Checking Float Graph...
2020-03-25 22:08:36.855342: W tensorflow/stream_executor/cuda/redzone_allocator.cc:312] Not found: ./bin/ptxas not found
Relying on driver to perform ptx compilation. This message will be only logged once.
INFO: Float Graph Check Done.
2020-03-25 22:08:38.228152: W tensorflow/contrib/decent_q/utils/quantize_utils.cc:538] Convert mean node resnet_v1_50/pool5 to AvgPool
2020-03-25 22:08:38.233940: W tensorflow/contrib/decent_q/utils/quantize_utils.cc:628] Scale output of avg_pool node resnet_v1_50/pool5 to simulate DPU.
INFO: Calibrating for 100 iterations...
100% (100 of 100) |###########################################################################################################################################################| Elapsed Time: 0:01:10 Time:  0:01:10
INFO: Calibration Done.
INFO: Generating Deploy Model...
[DEPLOY WARNING] Node resnet_v1_50/predictions/Reshape_1(Type: Reshape) is not quantized and cannot be deployed to DPU,because it has unquantized input node: resnet_v1_50/predictions/Softmax. Please deploy it on CPU.
INFO: Deploy Model Generated.
********************* Quantization Summary *********************
INFO: Output:
  quantize_eval_model: tf_resnetv1_50_imagenet_224_224_6.97G/vai_q_output/quantize_eval_model.pb
  deploy_model: tf_resnetv1_50_imagenet_224_224_6.97G/vai_q_output/deploy_model.pb
```

The **quantize_eval_model.pb** can be used for INT8 evaluation, DPU result emulation and compilation for DPUv3. The **deploy_model.pb** can be used for compilation for DPUv2. 

- ### Quantized model evaluation 

Run script **5_tf_eval_quantize_graph.sh** to evaluate Resnet50 quantized model. 

`sh 5_tf_eval_quantize_graph.sh`

Detailed command used in the script are as below:
```
EVAL_SCRIPT_PATH=example_file
EVAL_MODEL_PATH=vai_q_output

python ${EVAL_SCRIPT_PATH}/resnet_eval.py \
       --input_frozen_graph ${TF_NETWORK_PATH}/${EVAL_MODEL_PATH}/quantize_eval_model.pb  \
       --input_node input \
       --output_node resnet_v1_50/predictions/Reshape_1 \
       --eval_batches 100 \
       --batch_size 10 \
       --eval_image_dir images/ \
       --eval_image_list images/list.txt \
```

The evluation result is as below: 

```
22:11:34.201949: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1325] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 15022 MB memory) -> physical GPU (device: 0, name: Tesla V100-PCIE-16GB, pci bus id: 0000:3b:00.0, compute capability: 7.0)
22:11:34.203863: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x56449b87ce90 initialized for platform CUDA (this does not guarantee that XLA will be used). Devices:
2020-03-25 22:11:34.203877: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Tesla V100-PCIE-16GB, Compute Capability 7.0
N/A% (0 of 100) |                                                                                                                                                             | Elapsed Time: 0:00:00 ETA:  --:--:--22:11:37.252400: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcudnn.so.7
Relying on driver to perform ptx compilation. This message will be only logged once.
22:11:38.565843: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcublas.so.10.0
100% (100 of 100) |###########################################################################################################################################################| Elapsed Time: 0:00:05 Time:  0:00:05
Accuracy: 0.79
```

- ### Compilation for DPU edge overlay (DPUv2)
Run script **6_tf_compile_for_v2.sh** to compile Resnet50 deploy model for Vitis AI ZCU102 pre-built image. If other configuration is perferred, please refer to [UG1414 Vitis AI User Guide](https://www.xilinx.com/html_docs/vitis_ai/1_1/zkj1576857115470.html) about how to generate corresponding arch file and compile the model.

`sh 6_tf_compile_for_v2.sh`

Detailed commands used in the script are as below:

```
#DPUv2
TARGET=ZCU102
NET_NAME=resnet50
DEPLOY_MODEL_PATH=vai_q_output

ARCH=${CONDA_PREFIX}/arch/dpuv2/${TARGET}/${TARGET}.json

vai_c_tensorflow --frozen_pb ${TF_NETWORK_PATH}/${DEPLOY_MODEL_PATH}/deploy_model.pb \
                 --arch ${ARCH} \
                 --output_dir ${TF_NETWORK_PATH}/vai_c_output_${TARGET}/ \
                 --net_name ${NET_NAME} \
                 --options "{'save_kernel':''}"
```

After compilation is finished, compiler will report kernel information as below. 

```
**************************************************
* VITIS_AI Compilation - Xilinx Inc.
**************************************************
kernel list info for network "resnet50"
                               Kernel ID : Name
                                       0 : resnet50_0
                                       1 : resnet50_1

                             Kernel Name : resnet50_0
--------------------------------------------------------------------------------
                             Kernel Type : DPUKernel
                               Code Size : 0.64MB
                              Param Size : 24.35MB
                           Workload MACs : 6964.51MOPS
                         IO Memory Space : 2.25MB
                              Mean Value : 0, 0, 0,
                      Total Tensor Count : 59
                Boundary Input Tensor(s)   (H*W*C)
                              input:0(0) : 224*224*3

               Boundary Output Tensor(s)   (H*W*C)
         resnet_v1_50_logits_Conv2D:0(0) : 1*1*1000

                        Total Node Count : 58
                           Input Node(s)   (H*W*C)
            resnet_v1_50_conv1_Conv2D(0) : 224*224*3

                          Output Node(s)   (H*W*C)
           resnet_v1_50_logits_Conv2D(0) : 1*1*1000


                             Kernel Name : resnet50_1
--------------------------------------------------------------------------------
                             Kernel Type : CPUKernel
                Boundary Input Tensor(s)   (H*W*C)
        resnet_v1_50_SpatialSqueeze:0(0) : 1*1*1000

               Boundary Output Tensor(s)   (H*W*C)
   resnet_v1_50_predictions_Softmax:0(0) : 1*1*1000

                           Input Node(s)   (H*W*C)
             resnet_v1_50_SpatialSqueeze : 1*1*1000

                          Output Node(s)   (H*W*C)
        resnet_v1_50_predictions_Softmax : 1*1*1000

```

Important items (e.g. Kernel Name, Input Node(s) and Output Node(s)) will be used in programming. Generated elf file will be used in cross-compilation.

- ### Compilation for DPU Alveo overlay (DPUv3)
Run script **6_tf_compile_for_v3.sh** to compile Resnet50 deploy model for Vitis AI U50 DPUv3.

`sh 6_tf_compile_for_v3.sh`

Detailed commands used in the script are as below:

```
#DPUv3
ALVEO_TARGET=dpuv3e
NET_NAME=resnet50
DEPLOY_MODEL_PATH=vai_q_output

ALVEO_ARCH=/opt/vitis_ai/compiler/arch/${ALVEO_TARGET}/arch.json

vai_c_tensorflow --frozen_pb ${TF_NETWORK_PATH}/${DEPLOY_MODEL_PATH}/quantize_eval_model.pb \
                 --arch ${ALVEO_ARCH} \
                 --output_dir ${TF_NETWORK_PATH}/vai_c_output_${ALVEO_TARGET}/ \
                 --net_name ${NET_NAME}
```

After compilation is finished, compiler will report kernel information as below. 

```
**************************************************
* VITIS_AI Compilation - Xilinx Inc.
**************************************************
[INFO] ignore pretty_errors package
[INFO] Namespace(infer_shape=False, inputs_shape=None, layout='NHWC', model_files=['tf_resnetv1_50_imagenet_224_224_6.97G/vai_q_output/quantize_eval_model.pb'], model_type='tensorflow', out_filename='tf_resnetv1_50_imagenet_224_224_6.97G/vai_c_output_dpuv3e//resnet50_org.xmodel', proto=None)
[INFO] tensorflow model: tf_resnetv1_50_imagenet_224_224_6.97G/vai_q_output/quantize_eval_model.pb
[INFO] parse raw model :100%|######################################################################################################################################| 271/271 [00:00<00:00, 29806.90it/s]
[INFO] generate xmodel :100%|######################################################################################################################################| 220/220 [00:00<00:00, 548.71it/s]
[INFO] generate xmodel: /workspace/Tool-Example/tf_resnetv1_50_imagenet_224_224_6.97G/vai_c_output_dpuv3e/resnet50_org.xmodel
[UNILOG][INFO] The compiler log will be dumped at "/tmp/shuaizh/log/xcompiler-20200325-221923-1238"
[UNILOG][INFO] Target architecture: DPUv3e B4096
[UNILOG][INFO] Compile mode: dpu
[UNILOG][INFO] Begin to compile...
[UNILOG][INFO] Graph name: quantize_eval_model, with op num: 436
[UNILOG][INFO] Compile done.
[UNILOG][INFO] The meta json is saved to "/workspace/Tool-Example/tf_resnetv1_50_imagenet_224_224_6.97G/vai_c_output_dpuv3e/meta.json"
[UNILOG][INFO] The compiled xmodel is saved to "/workspace/Tool-Example/tf_resnetv1_50_imagenet_224_224_6.97G/vai_c_output_dpuv3e//resnet50.xmodel"
[UNILOG][INFO] The compiled xmodel's md5sum is 13ed55e10196dbab40e4c37cccf49aba, and been saved to "/workspace/Tool-Example/tf_resnetv1_50_imagenet_224_224_6.97G/vai_c_output_dpuv3e/md5sum.txt"
```

## 5. Summary

This section provides the examples of how to use Vitis AI toolchain to quantize, evaluate and compile the neural networks. For detailed information please refer to [UG1414 Vitis AI User Guide](https://www.xilinx.com/html_docs/vitis_ai/1_1/zkj1576857115470.html). 

Other useful documentations are listed below: 

[UG1354 Vitis AI Library User Guide](https://www.xilinx.com/support/documentation/sw_manuals/vitis_ai/1_1/ug1354-xilinx-ai-sdk.pdf)

[UG1355 Vitis AI Library Programming Guide](https://www.xilinx.com/support/documentation/sw_manuals/vitis_ai/1_1/ug1355-xilinx-ai-sdk-programming-guide.pdf)

[PG338 Zynq DPU IP Product Guide](https://www.xilinx.com/html_docs/vitis_ai/1_1/mgr1576863063269.html)

[UG1333 Vitis AI Optimizer User Guide](https://www.xilinx.com/html_docs/vitis_ai/1_1/thf1576862844211.html)
