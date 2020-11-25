/**
 * Copyright 2019 Xilinx Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
# Useful tools for board developing

## obtain fingerprint of DPU
```
xilinx_show_devices
```
## obtain fingerprint of Xmodel
```
xilinx_show_xmodel_kernel <xmodel> | grep fingerprint

```

## Obtain git commit versions of xir&vart&vitis-ai-library
```
```
## obtain inputs&outputs info
```
xilinx_show_xmodel_kernel <xmodel> | grep -A2 device=DPU
```

## xmodel to txt
```
```

## xmodel to png/svg
```
xilinx_xmodel_to_png <xmodel> <png>
xilinx_xmodel_to_svg <xmodel> <svg>
```

## show xmodel's kernels
```
xilinx_show_xmodel_kernel <xmodel>
```

## show device infos
```
xilinx_show_devices
```

## verify that load dpu.xclbin is normal
```
env [XLNX_VART_FIRMWARE=$bit_path/dpu.xclbin] xilinx_show_devices
```

## read register
```
```

## write register
```
```

## read data from device
```
```

## write data to device
```
```
