<table class="sphinxhide">
 <tr>
   <td align="center"><img src="https://raw.githubusercontent.com/Xilinx/Image-Collateral/main/xilinx-logo.png" width="30%"/><h1>Vitis AI</h1><h0>Adaptable & Real-Time AI Inference Acceleration</h0>
   </td>
 </tr>
</table>


# TensorFlow Lite with Vitis-AI Delegate

## 1. Cloud Deployment (amd64 host)

For cloud deployment, use the TensorFlow provided by Vitis AI:
```
conda activate vitis-ai-tensorflow
```

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

## 2. Edge Deployment (aarch64 host)

The edge deployment flow mirrors much of for the cloud with an additional step for cross-compiling for aarch64 on an amd64 host.  The cross-compilation flow exports a "rt_mod_holder.bin" file on the amd64 host that needs to be copied to the aarch64 host so it can import it upon edge deployment execution.

### A. Cross-compile model on amd64 host

Before we continue, ensure the following packages are installed on your amd64 host:
```
apt install gcc make gcc-aarch64-linux-gnu binutils-aarch64-linux-gnu
```

For cross-compiling on an amd64 host, use the TensorFlow provided by Vitis AI:
```
conda activate vitis-ai-tensorflow
```

#### A1. Compile libpyxir.so (amd64 host)

Compile libpyxir.so for an amd64 host (refer to the [PyXIR](https://github.com/Xilinx/pyxir) repo for more detailed instructions) and place it in ./dockerfiles/amd64/vitisai/pyxir/python overwriting the libpyxir.so file placeholder there.
```
git clone https://github.com/Xilinx/pyxir.git ./pyxir
cd pyxir
rm -rf build
git submodule update --init --recursive
python3 setup.py build_ext --debug --force
python3 setup.py install --user --debug
cd -
cp ./pyxir/python/libpyxir.so ./dockerfiles/amd64/vitisai/pyxir/python
```

#### A2. Compile libvitisai_delegate.so (amd64 host)

Compile libvitisai_delegate.so using the provided build script that uses a docker build environment.
```
cd ./dockerfiles/amd64
bash run.sh
cd -
```

#### A3. Execute Classification Example (amd64 host)

```
git clone https://github.com/tensorflow/examples ./examples
cd examples
git checkout -b tflite-vitisai-delegate 511baa3fbf7f049c853d2a4d43a5a75f3f548b96
git apply ../tflite-vitisai-delegate-examples.patch
cd -
cp ./dockerfiles/amd64/libvitisai_delegate.so ./examples/lite/examples/image_classification/raspberry_pi
cd ./examples/lite/examples/image_classification/raspberry_pi
PX_QUANT_SIZE=4 python classify_picamera.py --model vgg16.tflite --labels labels.txt --target DPUCZDX8G-zcu104
cd -
```

Copy the "rt_mod_holder.bin" file produced from the above steps onto the aarch64 host.  You will need to place it in the same ./examples/lite/examples/image_classification/raspberry_pi directory like on the amd64 host.

### B. Execute on aarch64 host

For executing on an aarch64 host you have the option of using the full TensorFlow package or the smaller TensorFlow Lite Runtime package for TensorFlow version 2.3.0.  If you choose to use the smaller TensorFlow Lite Runtime package, you will need to update the example "classify_picamera.py" file to import the corresponding python package, see [Run an inference using tflite_runtime](https://www.tensorflow.org/lite/guide/python).  For the full TensorFlow package, use the TensorFlow provided by pip:
```
pip install tensorflow=2.3.0
```

#### B1. Compile libpyxir.so (aarch64 host)

Compile libpyxir.so for an aarch64 host (refer to the [PyXIR](https://github.com/Xilinx/pyxir) repo for more detailed instructions) and place it in ./dockerfiles/aarch64/vitisai/pyxir/python overwriting the libpyxir.so file placeholder there.
```
git clone https://github.com/Xilinx/pyxir.git ./pyxir
cd pyxir
rm -rf build
git submodule update --init --recursive
python3 setup.py build_ext --debug --force --use_vai_rt_dpuczdx8g
python3 setup.py install --user --debug --use_vai_rt_dpuczdx8g
cd -
cp ./pyxir/python/libpyxir.so ./dockerfiles/aarch64/vitisai/pyxir/python
```

#### B2. Compile libvitisai_delegate.so (amd64 host)

Compile libvitisai_delegate.so using the provided build script that uses a docker build environment.
```
cd ./dockerfiles/aarch64
bash run.sh
cd -
```

Copy the "libvitisai_delegate.so" file produced from the above steps onto the aarch64 host.  For simplicity, you can place it in the same ./examples/lite/examples/image_classification/raspberry_pi directory like on the amd64 host.  Alternatively, you can place it in any other location and point to it with the "TFLITE_VITISAI_DELEGATE_PATH" environment variable.

#### B3. Execute Classification Example (aarch64 host)

```
git clone https://github.com/tensorflow/examples ./examples
cd examples
git checkout -b tflite-vitisai-delegate 511baa3fbf7f049c853d2a4d43a5a75f3f548b96
git apply ../tflite-vitisai-delegate-examples.patch
cd -
cp ./dockerfiles/aarch64/libvitisai_delegate.so ./examples/lite/examples/image_classification/raspberry_pi
cd ./examples/lite/examples/image_classification/raspberry_pi
PX_QUANT_SIZE=4 python classify_picamera.py --model vgg16.tflite --labels labels.txt --target DPUCZDX8G-zcu104
cd -
```
