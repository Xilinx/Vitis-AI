## Tensorflow SSD-Mobilenet Model

### Setup
```sh
# Activate Conda Environment
conda activate vitis-ai-caffe
```

### Data Preparation
- Download coco2014 datatset (https://cocodataset.org/#download)
> **Note:** User is responsible for the use of the downloaded content and compliance with any copyright licenses.

### Download xclbin
- Download and extract xclbin.tar.

### Running the Application
- `cd /workspace/demo/Whole-App-Acceleration/ssd_mobilenet/`
- `make build && make -j`
- `./run.sh <config file> <xmodel> <image path>`

