

## Running Caffe refinedet Models:
### Data Preparation
```
  1. Download coco2014 datatset 
  2. Reorg coco as below:
       coco2014/
             |->train2014/
             |->val2014/
             |->instances_train2014.json
             |->instances_val2014.json
  3. Convert coco to voc formate
     python convert_coco2voc_like.py
     outputs:
       coco2014/
             |->Annotations/
             |->Images/
             |->train2014.txt
             |->val2014.txt

  4. Generate ground truth file
     python generate_gt.py
```

```
# Setup ml-suite Environment Variables
source $VAI_ALVEO_ROOT/overlaybins/setup.sh

```

### Prepare a model for inference
** MUST BE DONE FIRST **
To run a Caffe model on the FPGA, it needs to be quantized, compiled, and a new graph needs to be generated. The new graph is similar to the original, with the FPGA subgraph removed, and replaced with a custom Python layer.

There are three versions of refinedet models are supported.
1. refinedet_pruned_0.8
2. refinedet_pruned_0.92
3. refinedet_pruned_0.96

```
cd $VAI_ALVEO_ROOT/examples/caffe/refinedet
python $VAI_ALVEO_ROOT/examples/caffe/getModels.py

//# Run the command below to generate compiled model for refinedet_pruned_0.8 
python run_refinedet.py --prototxt $VAI_ALVEO_ROOT/examples/caffe/models/refinedet_pruned_0.8/trainval.prototxt --caffemodel $VAI_ALVEO_ROOT/examples/caffe/models/refinedet_pruned_0.8/trainval.caffemodel --prepare

//# Run the command below to generate compiled model for refinedet_pruned_0.92 
python run_refinedet.py --prototxt $VAI_ALVEO_ROOT/examples/caffe/models/refinedet_pruned_0.92/trainval.prototxt --caffemodel $VAI_ALVEO_ROOT/examples/caffe/models/refinedet_pruned_0.92/trainval.caffemodel --prepare

//# Run the command below to generate compiled model for refinedet_pruned_0.96
python run_refinedet.py --prototxt $VAI_ALVEO_ROOT/examples/caffe/models/refinedet_pruned_0.96/trainval.prototxt --caffemodel $VAI_ALVEO_ROOT/examples/caffe/models/refinedet_pruned_0.96/trainval.caffemodel --prepare
```

### Run Inference for a single image
```
python run_refinedet.py --prototxt xfdnn_auto_cut_deploy.prototxt --caffemodel quantize_results/deploy.caffemodel --labelmap_file labelmap.prototxt --image <img_path>
```

### Run Inference on entire dataset and caluculate mAP
```
python run_refinedet.py --prototxt xfdnn_auto_cut_deploy.prototxt --caffemodel quantize_results/deploy.caffemodel --labelmap_file labelmap.prototxt --test_image_root ./coco2014/Images/ --image_list_file ./coco2014/val2014.txt --gt_file gt_file.txt --validate
```
