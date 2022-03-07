## 1. Directory description
```
root directory  
   |--Makefile: show how to generate XO files(VITIS RTL kernel files) and xclbin
   |--DPUCAHX8H_SRC
      |-DPU: (the DPU source codes in User-IP format when you want to deployment it on Alevo Card.
   |--release_$(card_type)_xo(_withdwc) (where to store XO source file and XO generation scripts)
   |--bit_gen: where to generate xclbin
      |-script: the configuration files needed by VITIS/Vivado
```

## 2. XO files
   User can utilize the Makefile to construct XOs with different configuration
   
### 2.1 Provided XO files
   | ENGINE_NUMBER |  ACLK_FREQ  |      CARD_TYPE     | 
   |---------------|-------------|--------------------|
   |   3ENGINE     |     300     |        u50         |
   |   5ENGINE     |     275     |       u50lv        |
   |   4ENGINE     |     275     |   u50lv_withDWC    |
   |  3,4,5ENGINE  |     250     |        u280        |
   |  3,4ENGINE    |     250     |   u280_withDWC     |
   |  3,4ENGINE    |     300     |        u55c        |
   
### 2.2 Generate XO with other configuration
   If you need other XO types, For examble: a lower frequency or less ENGINE_NUMBER
      a. Go into the path release_$(card_type)_xo(_withdwc);
      b. using the following command
      ```
       vivado -mode tcl  -source ./script/gen_DPUCAHX8H_ENGINE_xo.tcl -log ./log/vivado.log -jou ./log/vivado.jou -tclargs  ENGINE_NUMBER ACLK_FREQ TYPE
      ```
   There are three parameters can be configured in this IP

   **ENGINE_NUMBER:** The DPU ENGINE number. It is supported from 1ENGINE to 5ENGINE:
   
   **CLK_FREQ:** The DPU's clock frequency. The supported frequencies are: 300MHz, 275MHz, 250MHz, 225MHz, 200MHz and 100MHz. The value should be matched with the kernel clock. User can use below command in VITIS configuration file to set the kernel clock
   ```
   kernel_frequency=0:300|1:100 (set in cons_*.ini file)
   ```
   
   **CARD_TYPE:**      Currently, the supported card type include u50,u50lv,u50lv_withdwc,u280,u280_withdwc and u55c_withdwc  
                           
## 3. Generate bitstream(xclbin)
   At ./bit_gen, there is an example to show how to run VITIS to generate xclbin.
   ```
   make build ALVEO=u50 USE_2LV=true SLR0=4ENGINE SLR1=4ENGINE ACLK_FREQ=275 DWC_EN=1
   make build ALVEO=u55c USE_2LV=false SLR0=3ENGINE SLR1=4ENGINE SLR2=4ENGINE ACLK_FREQ=300 DWC_EN=1
   make build ALVEO=u280 USE_2LV=false SLR0=4ENGINE SLR1=5ENGINE SLR2=5ENGINE ACLK_FREQ=250 DWC_EN=0
   ```
   
   ***Noted:***
     1. Set  USE_2LV=true when you use u50lv card
     2. Set DWC_EN=1 when depthwise conv is enabled
     3. ACLK_FREQ in the command is used to set the kernel clock which is mentioned at 2.1.b
   
 **Please refer to PG367 for more detailed IP information**
