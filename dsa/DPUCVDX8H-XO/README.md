## 1. Directory description
```
root directory
├── 2pe_xo_release
│   ├── 2pe
│   │   ├── dpu.connection.cfg
│   │   ├── DPUCVDX8H.xo
│   │   └── libadf.a
│   └── 2pe.mk
├── 4pe_xo_release
│   ├── 4pe
│   │   ├── dpu.connection.cfg
│   │   ├── DPUCVDX8H.xo
│   │   └── libadf.a
│   └── 4pe.mk
├── 6pe_xo_release
│   ├── 6pe
│   │   ├── dpu.connection.cfg
│   │   ├── DPUCVDX8H.xo
│   │   └── libadf.a
│   └── 6pe.mk
└── 8pe_xo_release
    ├── 8pe
    │   ├── dpu.connection.cfg
    │   ├── DPUCVDX8H.xo
    │   └── libadf.a
    └── 8pe.mk
```
## 2. Generate bitstream(xclbin)
   There are 4 xo files provided in this repository with different AIE cores usage.
   
   |   PE_number   |  AIE_CORES_USAGE  | DWC_ENABLED | 
   |---------------|-------------------|-------------|
   |       2pe     |        128        |     Yes     |
   |       4pe     |        256        |     Yes     |
   |       6pe     |        384        |     Yes     |
   |       8pe     |        384        |     No      |
   
   User can ***change the frequency*** and ***assign the targeting platform*** in the ***makefile*** of each xo folder.
   Once finished, please use the following command to generate the xclbin.
   
   ```
   make xclbin
   ```
   Outputs will be saved in the directory: work/2021.2/
