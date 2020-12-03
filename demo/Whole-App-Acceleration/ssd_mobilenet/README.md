## Tensorflow SSD-Mobilenet Model

The mobilenet-ssd model is a Single-Shot multibox Detection (SSD) network intended to perform object detection. Accelerated post-processing(Sort and NMS) for ssd-mobilenet is provided and can only run on U280 board. In this application, software pre-process is used for loading input image, resize and mean subtraction.

### Setup
```sh
# Activate Conda Environment
conda activate vitis-ai-caffe
```

### Data Preparation
- Download coco2014 datatset (https://cocodataset.org/#download)
> **Note:** User is responsible for the use of the downloaded content and compliance with any copyright licenses.

### Download xclbin
- Download and extract xclbin tar. (XHD : `/proj/sdxapps/users/anup/public_downlaod/waa_system_u280_v1.3.0.tar.gz` or XSJ: `/wrk/acceleration/users/anup/public_download/waa_system_u280_v1.3.0.tar.gz`)
- `sudo tar -xf waa_system_u280_v1.3.0.tar.gz -C /usr/lib/`

### Running the Application
- `cd /workspace/demo/Whole-App-Acceleration/ssd_mobilenet/`
- `make build && make -j`
- Copy xmodel and config file from `/proj/sdxapps/users/anup/public_downlaod/model_ssd_mobilenet` or `/wrk/acceleration/users/anup/public_download/model_ssd_mobilenet`.
- `./run.sh model_ssd_mobilenet/ssd_mobilenet_v1_coco_tf.prototxt model_ssd_mobilenet/ssd_mobilenet_v1_coco_tf.xmodel <image path>`

