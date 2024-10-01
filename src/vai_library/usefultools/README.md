/**
 * Copyright 2022-2023 Advanced Micro Devices Inc.
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
python3 -m xdputil query  
```
## obtain fingerprint of Xmodel
```
python3 -m xdputil xmodel <xmodel> -l
```

## Obtain git commit versions of xir&vart&vitis-ai-library
```
python3 -m xdputil query  
```
## obtain inputs&outputs info
```
python3 -m xdputil xmodel <xmodel> -l
```

## xmodel to txt
```
python3 -m xdputil xmodel <xmodel> -t <TXT> 
```

## xmodel to png/svg
```
python3 -m xdputil xmodel <xmodel> -s <SVG> 
python3 -m xdputil xmodel <xmodel> -p <PNG> 
```

## show xmodel's kernels
```
python3 -m xdputil xmodel <xmodel> -l
```

## show device infos
```
python3 -m xdputil query  
```

## verify that load dpu.xclbin is normal
```
//?env [XLNX_VART_FIRMWARE=$bit_path/dpu.xclbin] xilinx_show_devices
```

## read register
```
python3 -m xdputil status
```

## read data from device
```
python3 -m xdputil [-r] addr size file
```

## write data to device
```
python3 -m xdputil [-w] addr size file
```
