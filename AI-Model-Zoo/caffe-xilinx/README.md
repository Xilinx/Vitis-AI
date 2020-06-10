# Caffe

[Caffe](https://github.com/BVLC/caffe) is a deep learning framework made with expression, speed, and modularity in mind.
It is developed by the Berkeley Vision and Learning Center ([BVLC](http://bvlc.eecs.berkeley.edu))
and community contributors.

# NVCaffe

[NVIDIA Caffe](https://github.com/NVIDIA/caffe) ([NVIDIA Corporation &copy;2017](http://nvidia.com)) is an NVIDIA-maintained fork
of BVLC Caffe tuned for NVIDIA GPUs, particularly in multi-GPU configurations.

# Xilinx Distribution of Caffe*

Xilinx Caffe ([Xilinx Corporation &copy;2019](http://www.xilinx.com)) is an XILINX-maintained fork of NVIDIA Caffe from branch caffe-0.15. Xilinx Caffe support FPGA friendly model quantization. After quantization, models can be deployed to FPGA devices. Xilinx Caffe is a component of [Xilinx Vitis AI](https://github.com/Xilinx/Vitis-AI), which is Xilinx’s development stack for AI inference on Xilinx hardware platforms.

## Building

Build procedure is the same as on bvlc-caffe-master branch. Both Make and CMake can be used. OpenMP will be used automatically if available.

## Running

Training and testing procedures are the same as on bvlc-caffe-master branch.
For quantization, refer to [Vitis AI](https://github.com/Xilinx/Vitis-AI).

## License and Citation

Caffe is released under the [BSD 2-Clause license](https://github.com/BVLC/caffe/blob/master/LICENSE).
The BVLC reference models are released for unrestricted use.

Please cite Caffe in your publications if it helps your research:

    @article{jia2014caffe,
      Author = {Jia, Yangqing and Shelhamer, Evan and Donahue, Jeff and Karayev, Sergey and Long, Jonathan and Girshick, Ross and Guadarrama, Sergio and Darrell, Trevor},
      Journal = {arXiv preprint arXiv:1408.5093},
      Title = {Caffe: Convolutional Architecture for Fast Feature Embedding},
      Year = {2014}
    }
