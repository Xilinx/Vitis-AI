## 1. Directory description
```
root directory  
   |--Makefile: show how to generate XO files(VITIS RTL kernel files) and xclbin
   |--DPUCAHX8H_SRC
      |-DPU: (the DPU source codes in User-IP format when you want to deployment it on Alevo Card.
   |--xo_release (where to store XO source file and XO generation scripts)
   |--bit_gen: where to generate xclbin
      |-script: the configuration files needed by VITIS/Vivado
```

## 2. Generate XO files
   User can utilize the Makefile to construct XOs with different configuration
   
### 2.1 Configuration
   There are three parameters can be configured in this IP

   **ENGINE_NUMBER:** The DPU ENGINE number. It is supported from 1ENGINE to 5ENGINE:
   
   **CLK_FREQ:** The DPU's clock frequency. The supported frequencies are: 300MHz, 275MHz and 250MHz. The value should be matched with the kernel clock. User can use below command in VITIS configuration file to set the kernel clock
   ```
   kernel_frequency=0:300|1:100 (set in cons_*.ini file)
   ```
   
   **CARD_TYPE:**      Currently, the supported card type includes u50,u50lv and u280
   
   ***Note:*** The default setting for the three types of card is as follows.
   
   | ENGINE_NUMBER |  ACLK_FREQ  |  CARD_TYPE  | 
   |---------------|-------------|-------------|
   |   3ENGINE     |     300     |     u50     |
   |   5ENGINE     |     275     |     u50lv   |
   |   4ENGINE     |     300     |     u280    |
   |   3ENGINE     |     300     |     u280    |
                            
### 2.2 Execution
   At root directory, execute one of the commands below to generate XO files for DPU with 0GB(_A) and 2GB(_B)
   ```
   make release_u50_xo
   make release_u50lv_xo
   make release_u280_xo
   ```
        
   Then, the generated XO locates at xo_release/DPUCAHX8H_*ENGINE.xoo

## 3. Generate bitstream(xclbin)
   At ./bit_gen, there is an example to show how to run VITIS to generate xclbin.
   ```
   make u50lv
   ```
   **cons_u50lv.ini:** VITIS configuration file to show how to control vivado and the memory-mapping between DPU and HBM
   
   **sys_link_u50lv.tcl:** show the physical AXI port connection between DPU and HBM.

**Please refer to PG367 for more detailed IP information**
