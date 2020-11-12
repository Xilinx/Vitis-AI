This directory containes examples for running Xilinx DPUCADX8G on Alveo platform. **DPUCADX8G**  is High Performance CNN processing engine designed for Xilinx Alveo-u200 and Alveo-u250 platforms. It was released as xDNNv3 as part of *ml-suite*. With **Vitis-AI**, Xilinx has integrated all the edge and cloud solutions under an unified API and toolset.

## Setup for Alveo
Targeting Alveo cards with Vitis-AI for AI workloads requires the installation of the following software components:  

* [Xilinx Run time(XRT)](https://github.com/Xilinx/XRT)  

* Alveo Deployment Shells (DSAs) - it  can be downloaded from Getting Started tab of Alveo board product page on [xilinx.com](xilinx.com)  
 
* Xilinx Resource Manager (XRM) (xbutler)  
 
* Xilinx Overlaybins (Accelerators to Dynamically Load – binary programming files) 
 
While it is possible to install all the software components individually, a script has been provided to automatically install them in one-shot. 

```sh
cd Vitis-AI/alveo/packages
# Run install script as root
sudo su
./install.sh
```
Then power cycle the system.
  
## Examples

 - [Jupyter Notebook Tutorials](notebooks/README.md)
   - [TensorFlow Image Classification](notebooks/image_classification_tensorflow.ipynb)
   - [Caffe Image Classification](notebooks/image_classification_caffe.ipynb)
   - [Caffe Object Detection w/ YOLOv2](notebooks/object_detection_yolov2.ipynb)
 - Command Line Examples
   - [TensorFlow ImageNet Benchmark Models](examples/tensorflow/README.md)
   - [Caffe ImageNet Benchmark Models](examples/caffe/README.md)
   - [Caffe VOC SSD Example](examples/caffe/ssd-detect/README.md)
   - [Deployment Mode Examples](examples/deployment_modes/README.md)

 ## Advanced Applications

 - [AI Kernel Scheduler (AKS)](apps/aks/README.md)
 - [Neptune](neptune/README.md)
 - [Whole App Acceleration](apps/whole_app_acceleration/README.md)
 - [Face Detect](apps/face_detect/README.md)

## References 
- [Performance Whitepaper][]
- **Watch:** [Webinar on Xilinx FPGA Accelerated Inference][] 


## Questions and Support
- [FAQ][]


[models]: docs/models.md
[Amazon AWS EC2 F1]: https://aws.amazon.com/marketplace/pp/B077FM2JNS
[Xilinx Virtex UltraScale+ FPGA VCU1525 Acceleration Development Kit]: https://www.xilinx.com/products/boards-and-kits/vcu1525-a.html
[AWS F1 Application Execution on Xilinx Virtex UltraScale Devices]: https://github.com/aws/aws-fpga/blob/master/SDAccel/README.md
[SDAccel Forums]: https://forums.xilinx.com/t5/SDAccel/bd-p/SDx
[Release Notes]: docs/release-notes/1.x.md
[UG1023]: https://www.xilinx.com/support/documentation/sw_manuals/xilinx2017_4/ug1023-sdaccel-user-guide.pdf
[FAQ]: docs/faq.md
[DPUCADX8G Overview]: docs/ml-suite-overview.md
[Webinar on Xilinx FPGA Accelerated Inference]: https://event.on24.com/wcc/r/1625401/2D3B69878E21E0A3DA63B4CDB5531C23?partnerref=Mlsuite
[ML Suite Forum]: https://forums.xilinx.com/t5/Xilinx-ML-Suite/bd-p/ML 
[ML Suite Lounge]: https://www.xilinx.com/products/boards-and-kits/alveo/applications/xilinx-machine-learning-suite.html
[Models]: https://www.xilinx.com/products/boards-and-kits/alveo/applications/xilinx-machine-learning-suite.html#gettingStartedCloud
[whitepaper here]: https://www.xilinx.com/support/documentation/white_papers/wp504-accel-dnns.pdf
[Performance Whitepaper]: https://www.xilinx.com/support/documentation/white_papers/wp504-accel-dnns.pdf
