## Re-identification

This code employs a resnet-18 based model to generate feature maps of person image. In order to identify same person or distinguish different persons, it uses cosine distances of feature maps to measure the distance between different images. The images having small distance are more likely to have same id, those having large distance are more likely to have different ids. When one query image and a set of gallery images are given, the image having the smallest distance to the query image will be regarded as the same person in query image. 


### Data Preparation

Download [market1501](http://www.diaochapai.com/survey/a61751ca-4210-4df1-a5bb-1e7a71b5262b) dataset. Extract the files to `./data/market1501`. The data structure should look like:
> **Note:** User is responsible for the use of the downloaded content and compliance with any copyright licenses.
```sh
data/
    market1501/
        query/
            xxx.jpg   
            xxx.jpg   
            ...
        bounding_box_train/
            xxx.jpg   
            xxx.jpg   
            ...
        bounding_box_test/
            xxx.jpg   
            xxx.jpg   
            ...
```

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

### Quantize, compile and generate subgraph prototxt

To run a caffe model on the FPGA, it needs to be quantized.

Quantize the model - The quantizer will generate scaling parameters for quantizing floats INT8. This is required, because FPGAs will take advantage of Fixed Point Precision, to achieve more parallelization at lower power

Compile the Model - In this step, the network files are compiled.

Subgraph Partitioning - In this step, the original graph is cut, and a custom FPGA accelerated python layer is inserted to be used for Inference.

```sh
python run.py --prototxt reid_model/trainval.prototxt --caffemodel reid_model/trainval.caffemodel --prepare
```

### Demo

The demo.py provides a running sample. 

```sh
python demo.py --query_image <query img_path> --test_image <test image_path> 
```

### Test Accuracy
```sh
python test_accuracy.py --img_dir <image dir>
```

```sh
usage: test_accuracy.py [-h] [--img_dir IMG_DIR]

optional arguments:
  -h, --help         show this help message and exit
  --img_dir IMG_DIR  Path to market1501 directory | default value: './data'
```
