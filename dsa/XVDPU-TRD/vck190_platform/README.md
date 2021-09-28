# VCK190 Base TRD

The Versal:tm: Base TRD consists of a series of platforms, accelerators, and Jupyter
notebooks targeting the VCK190 evaluation board. A platform is a Vivado:registered: design
with a pre-instantiated set of I/O interfaces and a corresponding PetaLinux BSP
and image that includes the required kernel drivers and user-space libraries to
exercise those interfaces. Accelerators are mapped to FPGA logic resources
and/or AI Engine cores and stitched into the platform using the Vitis:tm: unified software platform toolchain.

# Build Platform

To generate platform file, type the following command:
```bash
    make all
```
Note: It is important to note that Yocto/Petalinux requires the workspace(TMPDIR) can't be located on nfs.

The platform file .xpfm will be genereated at 'platforms/xilinx_vck190_mipiRxSingle_hdmiTx_202110_1/vck190_mipiRxSingle_hdmiTx.xpfm'

# License

Licensed under the Apache License, version 2.0 (the "License"); you may not use this file 
except in compliance with the License.

You may obtain a copy of the License at
[http://www.apache.org/licenses/LICENSE-2.0](http://www.apache.org/licenses/LICENSE-2.0)


Unless required by applicable law or agreed to in writing, software distributed under the 
License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, 
either express or implied. See the License for the specific language governing permissions 
and limitations under the License.    
