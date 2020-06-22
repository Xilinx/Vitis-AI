### Contents
1. [Installation](#installation)
2. [Preparation](#preparation)
3. [Train/Eval](#traineval)
4. [Performance](#performance)
5. [Model_info](#model_info)

### Installation

1. Environment requirement
    - pytorch 1.0

2. Quick installation
   ```shell
   conda env create -f code/configs/environment.yaml
   source activate facerec_env
   ```

### Preparation

1. prepare dataset.
   Please ensure that the data set has been face detected and face aligned.
   You could also get aligned faces according to caffe/facerec_resnet20/code/get_aligned_face.

2. Train dataset
   ```
   We used cleaned Microsoft dataset and Glint dataset.
   The number of class: 180855
   ```

3. Train dataset diretory structure
   ```
   + data
     + train
       + ms_glint
         + images
            + m.01r3tj5
                + 1.jpg
                + 2.jpg
                + ...
            + m.02qcsh6
                + ...
            + ...
         + lists
            + ms_glint.txt
   ```

4. Test dataset
   ```
   We used LFW dataset
   ```

5. Test dataset diretory structure
   ```
   + data
     + test
        + lfw
            + images
                + Joseph_Nacchio
                    + Joseph_Nacchio_0001.jpg
            + lfw.txt(List of all pictures)
            + pairs.txt
   ```

### Train/Eval

1. Train
    ```shell
    cd code/train
    bash train.sh
    ```
2. Eval
    ```shell
    cd code/test
    bash test.sh
    ```
3. Quantization
    ```shell
    cd code/quantize
    bash quant.sh
    ```
### Performance

|model|Accuracy|
|-|-|
|float|99.55%|

|model|Accuracy|
|-|-|
|INT8|99.52%|



### Model_info

1. Data preprocess
  ```
  data channel order: BGR(0~255)                  
  Image hegiht and width: h * w = 112* 96
  mean: 0.5
  std: 0.5
  output feature dim: 512D
  Loss: AM-softmax
  ``` 
