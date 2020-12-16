### Contents
1. [Installation](#installation)
2. [Preparation](#preparation)
3. [demo](#demo)
4. [Eval](#Eval)
5. [Performance](#performance)
6. [Model_info](#model_info)


### Installation
1. Get xilinx caffe-xilinx code. 
  ```shell
  # download or git clone caffe-xilinx
  unzip caffe-xilinx
  cd caffe-xilinx
  ```

2. Build the code. Please follow [Caffe instruction](http://caffe.berkeleyvision.org/installation.html) to install all necessary packages and build it.
  ```shell
  # Modify Makefile.config according to your Caffe installation.
  cp Makefile.config.example Makefile.config
  make -j8
  make pycaffe
  ```
3. Requirements
  ```shell
  #You may need to make sure all of the requirements are installed
  pip install -r requirements.txt
  ```
Note: If you are in the released Docker env, there is no need to build Caffe.

### Preparation

1. prepare dataset.
  
  test dataset: [WIDER Face](http://shuoyang1213.me/WIDERFACE/index.html)
  ```
  1.please download and extract WIDER FACE validation set to directory code/test/precisionTest/data/widerface
  2.Dataset Directory Structure like:
    + code/test/precisionTest/data
        + widerface
            + val
                + images
                + labels.txt
  ```
  
  test dataset: [FDDB](http://vis-www.cs.umass.edu/fddb/index.html), please download FDDB testSet images.  
  ```
  1.please download and extract FDDB dataset to directory code/test/precisionTest/data/fddb
  2.Dataset Directory Structure like:
    + code/test/precisionTest/data
        + fddb
            + FDDB_folds
            + images
            + test_result
  ```
  
2. prepare environment.

  ```shell
  cd code/test/precisionTest
  # make your retinaface tools compiled
  make

  cd evaluation/WiderFace-Evaluation
  # make Evaluation tool compiled
  python setup.py build_ext --inplace
  ``` 
    
### demo
1. Prepare test images

  You can randomly select several images from existing test dataset or use your own testset directly, or just skip this step (you will test with the world largest selfie)
  ```shell
  cd code/test/visualTest
  cp ../precisionTest/data/fddb/images/2002/08/26/big/img_265.jpg ./testImages/
  find testImages/ -name "*.jpg" > image_list_test.txt
  ```
2. run demo
  ```shell
  # check the path of picture, caffe and dataset, default setting is world_largest_selfie
  python test.py
  # You can see visualization output results from ./output.

  # if you test with quantized model, please add argument --nocrop(check the model dir and caffe dir)
  python test.py --nocrop
  ```

  
### Eval

1. Evaluate on FDDB dataset.
    
  FDDB testing requires the third-party evaluation tools, please download [evaluation tool](http://vis-www.cs.umass.edu/fddb/evaluation.tgz), unzip and compile it in directory code/test/precisionTest/evaluation/mkROC.

  If you have problems compiling it, you can refer to [FAQ](http://vis-www.cs.umass.edu/fddb/faq.html)
  ```shell
  cd code/test/precisionTest/
  # check and modify test_fddb.py to ensure paths are correct. 
  # evaluate
  python test_fddb.py
  # generate face detection result file in code/test/precisionTest/data/fddb/FDDB_results.txt

  #use the evaluation tool to generate ROC files. You can try to run the following command.
  evaluation/mkROC/evaluate -a $PATH/data/fddb/FDDB-folds/FDDB_annotations.txt -d $PATH/data/fddb/FDDB_results.txt -i $PATH/data/fddb/FDDB-folds/images/ -l $PATH/data/fddb/FDDB-folds/FDDB_list.txt -r $PATH/data/fddb/test_result -z .jpg
  
  # if you test with quantized model, please add argument --nocrop
  python test_fddb.py --nocrop
  ```  
  
2. Evaluate on WiderFace dataset.  
  ```shell
  cd code/test/precisionTest/
  # check and modify test_widerface.py to ensure paths are correct.
  # evaluate
  python test_widerface.py
  # Use evaluate tools to get AP on WiderFace
  cd evaluation/WiderFace-Evaluation
  #use evaluate tool to get AP
  python evaluation.py -p ../../wout -g ./
  ```

### Performance

|model|precision |Eval on FDDB| 
|----|----|----|
|float|Recall(%)|91.4@fp=100|
|quantized|Recall(%)|89.4@fp=100|

|Eval on WiderFace |Average Precision(float)|Average Precision(quantized)|
|----|----|----|
|easy|0.941|0.921|
|medium|0.919|0.893|
|hard|0.848|0.812|


### Model_info

1. data preprocess
  ```
  data channel order: BGR(0~255)
  padding: padding to image according to the ratio of 360:640(H:W)              
  resize: 360 * 640
  mean_value: 0, 0, 0
  scale: 1.0
  ```
2. For quantization with calibration mode:
   ```
   1. Replace the original network file data layer with the "ImageData" data layer
   2. Modify the "ImageData" layer parameters according to the data preprocess information
   3. Provide a "quant.txt" file, including image path and label information, but the label can randomly give some values
   4. Give examples of data layer and "quant.txt":

   # data layer example
     layer {
     name: "data"
     type: "ImageData"
     top: "data"
     top: "label"
     include {
       phase: TRAIN
     }
     transform_param {
       mirror: false
       mean_value: 0
       mean_value: 0
       mean_value: 0
      }
     image_data_param {
       source: "quant.txt"
       new_width: 640
       new_height: 360
       batch_size: 8
     }
   }
   # quant.txt: image path label
     images/000001.jpg 1
     images/000002.jpg 1
     images/000003.jpg 1
     …
```
   
