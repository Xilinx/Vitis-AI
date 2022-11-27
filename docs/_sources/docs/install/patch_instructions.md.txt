<table class="sphinxhide">
 <tr>
   <td align="center"><img src="https://raw.githubusercontent.com/Xilinx/Image-Collateral/main/xilinx-logo.png" width="30%"/><h1>Vitis AI</h1><h0>Adaptable & Real-Time AI Inference Acceleration</h0>
   </td>
 </tr>
</table> </br></br>


## Installing A Vitis AI Patch

Most Vitis AI components consist of Anaconda packages. These packages are distributed as tarballs, for example [unilog-1.3.2-h7b12538_35.tar.bz2](https://www.xilinx.com/bin/public/openDownload?filename=unilog-1.3.2-h7b12538_35.tar.bz2)

You can install the patches by starting the Vitis AI Docker container, and installing the package to a specific conda environment. For example, patching the `unilog` package in the `vitis-ai-caffe` conda environment:

```
Vitis-AI /workspace > cd /tmp
Vitis-AI /tmp > wget https://www.xilinx.com/bin/public/openDownload?filename=unilog-1.3.2-h7b12538_35.tar.bz2 -O unilog-1.3.2-h7b12538_35.tar.bz2
Vitis-AI /tmp > sudo conda install -n vitis-ai-caffe ./unilog-1.3.2-h7b12538_35.tar.bz2
```
