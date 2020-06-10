### Contents
1. [Installation](#installation)
2. [Preparation](#preparation)
3. [Train/Eval](#traineval)
4. [Performance](#performance)
5. [Model_info](#model_info)
6. [FAQ](#faq)

### Installation

1. Environment requirement
    - anaconda3
    - tensorflow 1.9+
    - opencv, tqdm, easydict etc. (Refer to [code/test/configs/environment.yaml](code/test/configs/environment.yaml) for more details.)

2. Quick installation
   ```shell
   # create environment
   conda env create -f code/test/configs/environment.yaml
   # activate the environment
   source activate tf_ssd_dev
   ```

### Preparation

1. Dataset description

The validation set is MSCOCO 14 minival set (note that our split is different from COCO 17 Val). A full list of image ids used in our split could be fould [here](https://github.com/tensorflow/models/blob/master/research/object_detection/data/mscoco_minival_ids.txt).

2. Download and prepare the dataset
    ```shell
    bash code/test/dataset_tools/prepare_data.sh
    ```

3. Dataset diretory structure
   ```
   + data
     + coco2014_minival_8059 
       + image
         + COCO_val2014_000000000042.jpg
         + COCO_val2014_000000000196.jpg
         + ...
       + minival2014_8059.txt
       + minival2014_8059.json
    ```

### Train/Eval

1. Evaluation
    Configure the model path and data path in [code/test/run_eval.sh](code/test/run_eval.sh)
    ```shell
    bash code/test/run_eval.sh
    ```

### Performance

|Model name           |Speed (ms) |Claimed mAP on COCO |Official ckpt mAP (score_thresh=0.005)|Our frozon pb mAP (score_thresh=0.005)|Official frozon pb mAP (score_thresh=0.3)|
|-                    |-          |-                   |-                                     |-                                    |-                                        |
|ssd_mobilenet_v1_coco     |30    |21       |20.8       |20.8       |17.8      |

Note: 
 - The accuracy of our frozon model(without preprocess&postprocess) is listed at the `Our frozon pb mAP` column. Refer to [the official detection modelzoo of tensorflow](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md) for more details.
 - If you try to evaluate the frozen graph, you may find performance numbers for some of the models to be slightly lower than what we report in the below tables. This is because we discard detections with scores below a threshold (typically 0.3) when creating the frozen graph. This corresponds effectively to picking a point on the precision recall curve of a detector (and discarding the part past that point), which negatively impacts standard mAP metrics.
 - Detector performance on subset of the COCO validation set or Open Images test split as measured by the dataset-specific mAP measure. Here, higher is better, and we only report bounding box mAP rounded to the nearest integer.

### Model_info

1. Data preprocess
  ```
  data channel order: RGB(0~255)                  
  resize: 300 * 300 (tf.image.resize_images(image, tf.stack([new_height, new_width]), method=tf.image.ResizeMethod.BILINEAR, align_corners=False))
  input = (2.0 / 255.0) * input - 1.0
  ``` 
2. Node information

  ```
  input node: 'image_tensor:0'
  output nodes: 'concat:0', 'concat_1:0'
  ```

### FAQ

Q1: How to export the object detection models without the preprocess and postprocess from tensorflow/models/research/object_detection?

A1: Refer to [code/test/export_tools/README.md](code/test/export_tools/README.md)

