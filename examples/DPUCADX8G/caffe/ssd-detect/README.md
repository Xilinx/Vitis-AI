
## Caffe Inception_v2 SSD Model

### Setup
> **Note:** Skip, If you have already run the below steps.

  Activate Conda Environment
  ```sh
  conda activate vitis-ai-caffe
  ```

  Setup the Environment

  ```sh
  source /vitis_ai_home/setup/alveo/u200_u250/overlaybins/setup.sh
  ```

### Data Preparation

Download VOC2007 dataset.
> **Note:** User is responsible for the use of the downloaded content and compliance with any copyright licenses.

```sh
# Download the data.
cd ${VAI_HOME}/examples/DPUCADX8G/caffe/ssd-detect
wget http://pjreddie.com/media/files/VOCtest_06-Nov-2007.tar
# Extract the data.
tar -xvf VOCtest_06-Nov-2007.tar
#Generate ground truth file
python generate_gt_file.py
```

>**:pushpin: NOTE:** VOC dataset contains 21 classes. But this model is trained with 19 classes (removed diningtable and train). If your model is having 21 classes, comment 46 line in generate_gt_file.py

The format of `calib.txt` used in calibration phase of `vai_q_caffe` is as follows:
```sh
# Image_name fake_label_number
000001.jpg 1
000002.jpg 1
000003.jpg 1
000004.jpg 1
000006.jpg 1
```
>**:pushpin: NOTE:** The label number is not actually used in calibration and arbitrary label number can be used.

```sh
# Get the necessary models
cd ${VAI_HOME}/examples/DPUCADX8G/caffe/ && python getModels.py && python replace_mluser.py --modelsdir models
```

### Prepare a model for inference

To run a Caffe model on the FPGA, it needs to be quantized, compiled, and a new graph needs to be generated. The new graph is similar to the original, with the FPGA subgraph removed, and replaced with a custom Python layer.

```sh
cd ${VAI_HOME}/examples/DPUCADX8G/caffe/ssd-detect

python run_ssd.py --prototxt ${VAI_HOME}/examples/DPUCADX8G/caffe/models/inception_v2_ssd/inception_v2_ssd_train.prototxt --caffemodel ${VAI_HOME}/examples/DPUCADX8G/caffe/models/inception_v2_ssd/inception_v2_ssd.caffemodel --prepare
```

### Run Inference on entire dataset and caluculate mAP
```sh
python run_ssd.py --prototxt xfdnn_auto_cut_deploy.prototxt --caffemodel quantize_results/deploy.caffemodel --labelmap_file labelmap_voc_19c.prototxt --test_image_root ./VOCdevkit/VOC2007/JPEGImages/ --image_list_file ./VOCdevkit/VOC2007/ImageSets/Main/test.txt --gt_file voc07_gt_file_19c.txt --validate
```

### Run Inference for a single image
```sh
python run_ssd.py --prototxt xfdnn_auto_cut_deploy.prototxt --caffemodel quantize_results/deploy.caffemodel --labelmap_file labelmap_voc_19c.prototxt --image Yogi.jpeg
```
