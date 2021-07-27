## 1. Directory Description
   ```
   root directory
   |--Makefile: show how to generate XO files(VITIS RTL kernel files) and xclbin
   |--DPUCAHX8L_A_SRC
      |-DPU: (the _A DPU source codes in User-IP format when you want to deployment it on Alevo Card. It use address-offset 0GB in HBM memory system. )
   |--DPUCAHX8L_B_SRC
      |-DPU: (the _B DPU source codes in User-IP format when you want to deployment it on Alevo Card. It use address-offset 2GB in HBM memory system. )
   |--xo_release
      |-VB_0GB: (where to store 0GB XO file and XO generation scripts. here, 0GB use sources codes from DPUCAHX8L_A_SRC)
      |-VB_2GB: (where to store 2GB XO file and XO generation scripts. here, 2GB use sources codes from DPUCAHX8L_B_SRC)
   |--bit_gen: where to generate xclbin
      |-script: the configuration files needed by VITIS/Vivado
  ```


## 2. Generate XO files
   User can use Makefile to construct XOs with different configuration
   
   ### 2.1 configuration
   For DPUv3me, only the frequency can be configured
   
   ACLK_FREQ: the DPU's clock frequency. the supported value are: 300MHz, 275MHz and 250MHz(default). This value should match the kernel clock.
   
   you can use below command in VITIS configuration file to set the kernel clock
   
   ```bash
   kernel_frequency=0:300|1:100
   ```
   
   **Note**: 
   
   1. in the <root diretory>/bit_gen/script/cons.ini, we use a default 300MHz as an example.
  
   2. if you select U50LV as ALVEO board, please choose 275MHz or 250MHz.

   ### 2.2 run
   At root directory use below commands to generate XO files for DPU with 0GB(_A) and 2GB(_B)
   
   ```bash
   make release_DPUCAHX8L_A_xo
   make release_DPUCAHX8L_B_xo
   ```
   
  Then, the generated XO locates at xo_release/VB_0GB/DPUCAHX8L_A.xo and xo_release/VB_2GB/DPUCAHX8L_B.xo

## 3. Generate bitstream(xclbin)
At ./bit_gen, there is an example to show how to run VITIS to generate xclbin.

```bash
make u280_1core
```

**Note**: 
1. cons_u280_1core.ini: VITIS configuration file to show how to control vivado and the memory-mapping between DPU and HBM

2. sys_link_post.tcl: show the physical AXI port connection between DPU and HBM. It is design requirement because it can realize the function in 4.1.1

3. opt_design.pre.tcl: during Vivado implementation flow, there will be an error because there is MMCM-cascade. Ues this file to avoid it

## 4. Important
**The differences between _A and _B:**

   _A use 0GB as address-offset in HBM memory system and _B use 2GB as address-offset in HBM memory system. The address-offset is used by DPU internal for best performance.
   
   the required placement for 2 cores
  |        Board            | which core to be selected | suggested SLR assignment  |
  |-------------------------|---------------------------|---------------------------|
  |        U50/U50LV        | only _A can be used       |    _A in SLR1 only        |
  |        U280             |      _A is used           | _A in SLR1 only           |
  |        U280             | _A and _B are used        | _A in SLR1 and _B in SLR2 |

**Please use XRT later than 2020.2**
   
**Please refer to PG366 for more detailed IP information**
