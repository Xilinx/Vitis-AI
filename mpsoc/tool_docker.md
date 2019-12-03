# Vitis AI Tool Docker

## 1. Prerequisite
- Docker service is available on host machine. For detailed steps, please refer to [Install Docker](/doc/install_docker/README.md) page
- Vitis AI CPU/GPU docker image is correctly loaded and launched. For detailed steps, please refer to [Load and Run Docker](/doc/install_docker/load_run_docker.md)
- All necessary files are placed in `/workspace` folder 

## 2. File Preparation

Caffe and TensorFlow Resnet50 are used in following examples. Network packages can be found on [Xilinx AI Model Zoo](https://github.com/Xilinx/AI-Model-Zoo). 

Refernece files used in the examples can be found on [Workflow Example](https://github.com/Xilinx/Vitis-AI/tree/master/mpsoc/vitis-ai-tool-example). After downloading the model packages (No. 1 and No. 6) from AI Model Zoo, please unzip them under `vitis-ai-tool-example/` and rename model folders as ***cf_resnet50*** and ***tf_resnet50*** respectively.

In order to conduct calibration and evaluation in quantization phase, please download pictures and labels from ImageNet and place them under `vitis-ai-tool-example/images/`. *ILSVRC2012_val_00000001.JPEG* to *ILSVRC2012_val_00001000.JPEG* are used as example. 

## 3. Caffe Workflow

Network file structure used in this example is shown as below:

```
Vitis-AI-Tool-Example/cf_resnet50/
|-- deploy
|   |-- deploy.caffemodel
|   `-- deploy.prototxt
|-- fix
|   |-- deploy.caffemodel
|   |-- deploy.prototxt
|   |-- fix_test.prototxt
|   |-- fix_train_test.caffemodel
|   `-- fix_train_test.prototxt
|-- float
|   |-- float.caffemodel
|   |-- float.prototxt
|   |-- test.prototxt
|   `-- trainval.prototxt
`-- readme.md
```

Before started, please use **0_set_env.sh** to set several necessary parameters, such as path to the networks. 

`. 0_set_env.sh`

- ### Quantization

In order to use Caffe quantization tool (**vai_q_caffe**), please use following command to activate caffe tool conda environment

`conda activate vitis-ai-caffe`

Run script **1_caffe_quantize.sh** to calibrate, quantize and evaluate Resnet50. 

`sh 1_caffe_quantize.sh`

Detailed command used in the script are as below:

```
vai_q_caffe quantize -model ${CF_NETWORK_PATH}/float/trainval.prototxt \
                     -weights ${CF_NETWORK_PATH}/float/float.caffemodel \
                     -output_dir ${CF_NETWORK_PATH}/vai_q_output \
                     -calib_iter 100 \
                     -test_iter 100 \
                     -auto_test \
```

Main output are as below:

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
Output Quantized Train&Test Model:   "cf_resnet50/vai_q_output/quantize_train_test.prototxt"
Output Quantized Train&Test Weights: "cf_resnet50/vai_q_output/quantize_train_test.caffemodel"
Output Deploy Weights: "cf_resnet50/vai_q_output/deploy.caffemodel"
Output Deploy Model:   "cf_resnet50/vai_q_output/deploy.prototxt"
```

The **quantize_train_test** model can be used for INT8 evaluation, finetune or DPU result emulation. The **deploy** model can be used to generate DPU elf file. 


- ### Compilation

Run script **2_caffe_compile.sh** to compile Resnet50 deploy model for Vitis AI ZCU102 pre-built image. 

`sh 2_caffe_compile.sh`

Detailed command used in the script are as below:

```
TARGET=ZCU102
NET_NAME=resnet50
DEPLOY_MODEL_PATH=vai_q_output
ARCH=/opt/vitis_ai/compiler/arch/dpuv2/${TARGET}/${TARGET}.json

vai_c_caffe --prototxt ${CF_NETWORK_PATH}/${DEPLOY_MODEL_PATH}/deploy.prototxt \
            --caffemodel ${CF_NETWORK_PATH}/${DEPLOY_MODEL_PATH}/deploy.caffemodel \
            --arch ${ARCH} \
            --output_dir ${CF_NETWORK_PATH}/vai_c_output_${TARGET}/ \
            --net_name ${NET_NAME} \

```

After compilation is finished, compiler will report kernel information as below. 

```
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

## 4. TensorFlow Workflow

Network file structure used in this example is shown as below:

```
Vitis-AI-Tool-Example/tf_resnet50/
|-- fix
|   |-- deploy_model.pb
|   `-- quantize_eval_model.pb
|-- float
|   `-- frozen.pb
|-- input_fn.py
`-- readme.md
```

Due to the nature of TensorFlow, the input function and evaluation scripts need to be prepared for networks repectively. The files used in this example are listed in **tf_eval_script**.

Before started, please use **0_set_env.sh** to set several necessary parameters, such as path to the networks. 

`. 0_set_env.sh`

In order to use TensorFlow and quantization tool (**vai_q_tensorflow**), please use following command to activate TensorFlow tool conda environment

`conda activate vitis-ai-tensorflow`

- ### Float model evaluation 

Run script **3_tf_eval_frozen_graph.sh** to evaluate Resnet50 float model. 

`sh 3_tf_eval_frozen_graph.sh`

Detailed command used in the script are as below:
```
EVAL_SCRIPT_PATH=tf_eval_script

python ${EVAL_SCRIPT_PATH}/resnet_eval.py \
       --input_frozen_graph ${TF_NETWORK_PATH}/float/frozen.pb \
       --input_node input \
       --output_node resnet_v1_50/predictions/Reshape_1 \
       --eval_batches 100 \
       --batch_size 10 \
       --eval_image_dir images/ \
       --eval_image_list images/list.txt \
```

The evluation result is as below: 

```
2019-11-28 19:22:59.957137: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1511] Adding visible gpu devices: 0
2019-11-28 19:23:00.310842: I tensorflow/core/common_runtime/gpu/gpu_device.cc:982] Device interconnect StreamExecutor with strength 1 edge matrix:
2019-11-28 19:23:00.310870: I tensorflow/core/common_runtime/gpu/gpu_device.cc:988]      0
2019-11-28 19:23:00.310874: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1001] 0:   N
2019-11-28 19:23:00.310987: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1115] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 14943 MB memory) -> physical GPU (device: 0, name: Tesla V100-PCIE-16GB, pci bus id: 0000:3b:00.0, compute capability: 7.0)
100% (100 of 100) |###############################################################################################################| Elapsed Time: 0:00:03 Time:  0:00:03
Accuracy: 0.77
```


- ### Quantization

Run script **4_tf_quantize.sh** to calibrate and quantize Resnet50. 

`sh 4_tf_quantize.sh`

Detailed command used in the script are as below:

```
vai_q_tensorflow quantize --input_frozen_graph ${TF_NETWORK_PATH}/float/frozen.pb \
                          --input_fn ${TF_NETWORK_PATH}.input_fn.calib_input \
                          --output_dir ${TF_NETWORK_PATH}/vai_q_output \
                          --input_nodes input \
                          --output_nodes resnet_v1_50/predictions/Reshape_1 \
                          --input_shapes ?,224,224,3 \
                          --calib_iter 50 \
```

Main output are as below:

```
INFO: Checking Float Graph...
INFO: Float Graph Check Done.
2019-11-28 19:32:36.976444: W tensorflow/contrib/decent_q/utils/quantize_utils.cc:570] Convert mean node resnet_v1_50/pool5 to AvgPool
2019-11-28 19:32:36.985805: W tensorflow/contrib/decent_q/utils/quantize_utils.cc:660] Scale output of avg_pool node resnet_v1_50/pool5 to simulate DPU.
INFO: Calibrating for 50 iterations...
100% (50 of 50) |#################################################################################################################| Elapsed Time: 0:00:35 Time:  0:00:35
INFO: Calibration Done.
INFO: Generating Deploy Model...
INFO: Deploy Model Generated.
********************* Quantization Summary *********************
INFO: Output:
  quantize_eval_model: tf_resnet50/vai_q_output/quantize_eval_model.pb
  deploy_model: tf_resnet50/vai_q_output/deploy_model.pb

```

The **quantize_eval_model.pb** can be used for INT8 evaluation and DPU result emulation. The **deploy_model.pb** can be used to generate DPU elf file. 

- ### Quantized model evaluation 

Run script **5_tf_eval_quantize_graph.sh** to evaluate Resnet50 quantized model. 

`sh 5_tf_eval_quantize_graph.sh`

Detailed command used in the script are as below:
```
EVAL_SCRIPT_PATH=tf_eval_script
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
2019-11-28 19:35:11.761955: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1511] Adding visible gpu devices: 0
2019-11-28 19:35:12.111228: I tensorflow/core/common_runtime/gpu/gpu_device.cc:982] Device interconnect StreamExecutor with strength 1 edge matrix:
2019-11-28 19:35:12.111259: I tensorflow/core/common_runtime/gpu/gpu_device.cc:988]      0
2019-11-28 19:35:12.111264: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1001] 0:   N
2019-11-28 19:35:12.111390: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1115] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 14943 MB memory) -> physical GPU (device: 0, name: Tesla V100-PCIE-16GB, pci bus id: 0000:3b:00.0, compute capability: 7.0)
100% (100 of 100) |###############################################################################################################| Elapsed Time: 0:00:05 Time:  0:00:05
Accuracy: 0.79
```

- ### Compilation
Run script **6_tf_compile.sh** to compile Resnet50 deploy model for Vitis AI ZCU102 pre-built image. 

`sh 6_tf_compile.sh`

Detailed command used in the script are as below:

```
TARGET=ZCU102
NET_NAME=resnet50
DEPLOY_MODEL_PATH=vai_q_output
ARCH=/opt/vitis_ai/compiler/arch/dpuv2/${TARGET}/${TARGET}.json

vai_c_tensorflow --frozen_pb ${TF_NETWORK_PATH}/${DEPLOY_MODEL_PATH}/deploy_model.pb \
                 --arch ${ARCH} \
                 --output_dir ${TF_NETWORK_PATH}/vai_c_output_${TARGET}/ \
                 --net_name ${NET_NAME}
```

After compilation is finished, compiler will report kernel information as below. 

```
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
