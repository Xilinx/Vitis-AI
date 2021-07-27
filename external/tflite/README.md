# TensorFlow Lite with Vitis-AI Delegate

## 1. Cloud Deployment (amd64 host)

### 1. Compile libpyxir.so

Compile libpyxir.so for an amd64 host (refer to the [PyXIR](https://github.com/Xilinx/pyxir) repo for more detailed instructions) and place it in ./dockerfiles/amd64/vitisai/pyxir/python overwriting the libpyxir.so file placeholder there.
```
git clone https://github.com/Xilinx/pyxir.git ./pyxir
cd pyxir
rm -rf build
git submodule update --init --recursive
python3 setup.py build_ext --debug --force --use_vai_rt_dpucahx8h
python3 setup.py install --user --debug --use_vai_rt_dpucahx8h
cd -
cp ./pyxir/python/libpyxir.so ./dockerfiles/amd64/vitisai/pyxir/python
```

### 2. Compile libvitisai_delegate.so

Compile libvitisai_delegate.so using the provided build script that uses a docker build environment.
```
cd ./dockerfiles/amd64
bash run.sh
cd -
```

### 3. Execute Classification Example

```
git clone https://github.com/tensorflow/examples ./examples
cd examples
git checkout -b tflite-vitisai-delegate 511baa3fbf7f049c853d2a4d43a5a75f3f548b96
git apply ../tflite-vitisai-delegate-examples.patch
cd -
cp ./dockerfiles/amd64/libvitisai_delegate.so ./examples/lite/examples/image_classification/raspberry_pi
cd ./examples/lite/examples/image_classification/raspberry_pi
PX_QUANT_SIZE=4 python classify_picamera.py --model vgg16.tflite --labels labels.txt --target DPUCAHX8H-u50
cd -
```
