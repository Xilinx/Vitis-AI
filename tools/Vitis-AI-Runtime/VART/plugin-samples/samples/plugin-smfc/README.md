# SoftMax IP Plugin for edge devices

## Build 

```sh
# Setup env
conda activate vitis-ai-tensorflow

# Go to the plugin-samples directory
cd /workspace/tools/Vitis-AI-Runtime/VART/plugin-samples

# Build plugin
./cmake.sh --clean
```
This build the plugin libraries under `~/.local/Ubuntu.18.04.x86_64.Debug/lib`.

## Compile Model with Plugin (ResNet50)

```sh
# Set the path to the plugin library
export LD_LIBRARY_PATH=~/.local/Ubuntu.18.04.x86_64.Debug/lib

# Download the Quatized model from model zoo
wget https://www.xilinx.com/bin/public/openDownload?filename=tf_resnetv1_50_imagenet_224_224_6.97G_1.3.zip -O tf_resnetv1_50_imagenet_224_224_6.97G_1.3.zip

# Unzip
unzip tf_resnetv1_50_imagenet_224_224_6.97G_1.3.zip

# Compile the model
vai_c_tensorflow -a /opt/vitis_ai/compiler/arch/DPUCZDX8G/ZCU102/arch.json -f tf_resnetv1_50_imagenet_224_224_6.97G_1.3/quantized/quantize_eval_model.pb -o xmodel -n resnet_v1_50 --options '{"plugins": "plugin-smfc"}'
```

## Example application (ResNet50)

After the `xmodel` is generated, we can run the compiled `xmodel` on ZCU102 using VART. As [VART User Guide](https://github.com/Xilinx/Vitis-AI/blob/master/tools/Vitis-AI-Runtime/VART/quick_start_for_edge.md) introduces take these steps to deploy this model on ZCU102 board, setting up the host, setting up the target and running vitis ai examples.

- Compile Example
    ```sh
    # Compile the example
    cd /workspace/tools/Vitis-AI-Runtime/VART/vart/softmax-runner/test/resnet_v1_50_softmax
    ./build.sh
    ```
- Run Example
    ```sh
    mkdir ~/resnet50
    [Host]scp resnet_v1_50_softmax <path to compiled model>/resnet_v1_50.xmodel root@[IP_OF_BOARD]:~/resnet50/
    cp ~/Vitis-AI/examples/VART/samples/images/001.jpg ~/resnet50/
    cd  ~/resnet50
    ./resnet_v1_50_softmax resnet_v1_50.xmodel 001.jpg
    ```

