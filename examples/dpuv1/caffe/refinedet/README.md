## Caffe refinedet Models

### Setup
> **Note:** Skip, If you have already run the below steps.

  Activate Conda Environment
  ```sh
  conda activate vitis-ai-caffe 
  ```

  Setup the Environment

  ```sh
  source /workspace/alveo/overlaybins/setup.sh
  ```

### Data Preparation

- Download coco2014 datatset (https://cocodataset.org/#download) 
> **Note:** User is responsible for the use of the downloaded content and compliance with any copyright licenses.
- Reorg coco as below:
  ```sh
  coco2014/
        |->train2014/
        |->val2014/
        |->instances_train2014.json
        |->instances_val2014.json
  ```
- Convert coco to voc format
  ```sh
  python convert_coco2voc_like.py
  outputs:
    coco2014/
          |->Annotations/
          |->Images/
          |->train2014.txt
          |->val2014.txt
  ```
- Generate ground truth file
  ```sh
  python generate_gt.py
  ```


### Prepare models for inference

To run a Caffe model on the FPGA, it needs to be quantized, compiled, and a new graph needs to be generated. The new graph is similar to the original, with the FPGA subgraph removed, and replaced with a custom Python layer.

Three versions of refinedet models are supported.
1. refinedet_pruned_0.8
2. refinedet_pruned_0.92
3. refinedet_pruned_0.96

Get the necessary models
  ```sh
  cd ${VAI_ALVEO_ROOT}/examples/caffe
  python getModels.py
  python replace_mluser.py --modelsdir models
  cd ${VAI_ALVEO_ROOT}/examples/caffe/refinedet
  ```

```sh
# Run the command below to generate compiled model for refinedet_pruned_0.8 

python run_refinedet.py --prototxt ${VAI_ALVEO_ROOT}/examples/caffe/models/refinedet_pruned_0.8/trainval.prototxt --caffemodel ${VAI_ALVEO_ROOT}/examples/caffe/models/refinedet_pruned_0.8/trainval.caffemodel --prepare

```

```sh
# Run the command below to generate compiled model for refinedet_pruned_0.92 

python run_refinedet.py --prototxt ${VAI_ALVEO_ROOT}/examples/caffe/models/refinedet_pruned_0.92/trainval.prototxt --caffemodel ${VAI_ALVEO_ROOT}/examples/caffe/models/refinedet_pruned_0.92/trainval.caffemodel --prepare

```

```sh
# Run the command below to generate compiled model for refinedet_pruned_0.96

python run_refinedet.py --prototxt ${VAI_ALVEO_ROOT}/examples/caffe/models/refinedet_pruned_0.96/trainval.prototxt --caffemodel ${VAI_ALVEO_ROOT}/examples/caffe/models/refinedet_pruned_0.96/trainval.caffemodel --prepare
```

### Run Inference for a single image
```sh
python run_refinedet.py --prototxt xfdnn_auto_cut_deploy.prototxt --caffemodel quantize_results/deploy.caffemodel --labelmap_file labelmap.prototxt --image <img_path>
```

### Run Inference on entire dataset and caluculate mAP
```sh
python run_refinedet.py --prototxt xfdnn_auto_cut_deploy.prototxt --caffemodel quantize_results/deploy.caffemodel --labelmap_file labelmap.prototxt --test_image_root ./coco2014/Images/ --image_list_file ./coco2014/val2014.txt --gt_file gt_file.txt --validate
```
