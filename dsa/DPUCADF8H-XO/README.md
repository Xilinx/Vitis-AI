## 1. Directory description
```
ml_shell directory structure

├── Makefile                                                      # Makefile      
├── outputs                                                       # Directory to save checkpoints, reports, etc.    
  ├── checkpoints                                                 # Directory to save checkpoints
  ├── logs                                                        # Vivado.log      
  ├── reports                                                     # Directory to save reports         
  └── xclbin                                                      # Directory to save xclbin        
├── scripts                                                       # Implementaion flow scripts, constraints
  ├── proc_tcl                                                    # Process bert pl filelist for Vivado           
    └── proc_vivado.tcl                                           # Write reports, checkpoints                                                                  
  ├── syn_tcl                                                     # Implementation scripts          
    ├── gen_ips.tcl                                               # Generate IPs for PL(Not used)                  
    ├── gen_xo.tcl                                                # Generate XO packages                 
    ├── link_post.tcl                                             # Post link script                    
    ├── opt_post.tcl                                              # Post opt script                   
    ├── opt_pre.tcl                                               # Pre opt script                  
    ├── package_kernel.tcl                                        # Package kernel to XO packages                         
    ├── place_post.tcl                                            # Post place script                    
    ├── place_pre.tcl                                             # Pre place script                    
    ├── route_post.tcl                                            # Post route script                     
    ├── route_pre.tcl                                             # Pre route script                                                              
    ├── synth_post.tcl                                            # Post synth script                     
    └── synth_pre.tcl                                             # Pre synth script                                                              
  ├── user_setup                                                  #              
    ├── env_config.tcl                                            # Config variables in implementation                    
    └── user_setup.tcl                                            # Set variables in implementation                                         
├── src                                                           # Sources                        
  ├── constraints                                                 # Directory for timing and physical constraints              
    ├── physical                                                  # Physical constraints               
      ├── xilinx_u200_xdma_201830_2_slr0_physical.tcl             # U200 slr0 xdma_201830_2 physical constraints
      ├── xilinx_u200_xdma_201830_2_slr2_physical.tcl             # U200 slr2 xdma_201830_2 physical constraints
      ├── xilinx_u200_gen3x16_xdma_base_1_slr0_physical.tcl       # U200 slr0 gen3x16_xdma_base_1 physical constraints
      ├── xilinx_u200_gen3x16_xdma_base_1_slr2_physical.tcl       # U200 slr2 gen3x16_xdma_base_1 physical constraints
      ├── xilinx_u250_xdma_201830_2_slr0_physical.tcl             # U250 slr0 xdma_201830_2 physical constraints
      ├── xilinx_u250_xdma_201830_2_slr1_physical.tcl             # U250 slr1 xdma_201830_2 physical constraints
      ├── xilinx_u250_xdma_201830_2_slr2_physical.tcl             # U250 slr2 xdma_201830_2 physical constraints
      ├── xilinx_u250_xdma_201830_2_slr3_physical.tcl             # U250 slr3 xdma_201830_2 physical constraints
      ├── xilinx_u250_gen3x16_xdma_3_1_slr0_physical.tcl          # U250 slr0 gen3x16_xdma_3_1 physical constraints
      ├── xilinx_u250_gen3x16_xdma_3_1_slr2_physical.tcl          # U250 slr2 gen3x16_xdma_3_1 physical constraints
      ├── xilinx_u250_gen3x16_xdma_3_1_slr3_physical.tcl          # U250 slr3 gen3x16_xdma_3_1 physical constraints
    ├── property                                                  # Config vivado implementation parameters               
      └── vivado_property.tcl                                     #                           
    └── timing                                                    #              
      └── dpdpuv3_wrapper.tcl                                     # Timing constraints                               
  └── kernel.xml                                                  # kernel.xml             

```

## 2. Generate XO files
   User can enter the ***ml_shell*** directory, utilize the Makefile to construct XOs with different configuration
   ```
   cd project/ml_shell/
   vim Makefile
   ```
   
### 2.1 Configuration
   There are three parameters can be configured in this IP
   
   **BOARD_TYPE:** Currently, the supported board type includes u200 and u250
   
   **PLATFORM:** The platform name. Ideally, it supports all platforms on u200 and u250 cards. The following platforms are the ones have been validated.
   ```
   xilinx_u200_xdma_201830_2
   xilinx_u200_gen3x16_xdma_base_1
   xilinx_u250_xdma_201830_2
   xilinx_u250_gen3x16_xdma_3_1
   ```
   
   **CLK_FREQ:** The clock is internal to the DPU. The supported frequency includes 50,400,500 and 600. It is recommended to set to 600.
                            
### 2.2 Execution
   Execute the commands below to generate XO files for DPU
   ```
   make gen_xo
   ```
        
   Then, the generated XO locates at projects/ml_shell/_x.hw.xilinx_<BOARD>_<PLATFORM>.enc_true/DPUCADF8H.xo

## 3. Generate bitstream(xclbin)
   Except the configuration above, user also needs to configure the following parameters.

   **DPU_SLR_IDX:** It is an array, it means the  SLR location of each DPU. For example, the "0 2" means there are 2 DPU, one is in SLR0 and the other one is in SLR2. 
   
   ***Noted:*** U200 and U250 platform gen3x16_xdma_3_1 can not use SLR1 for DPU because of the shape of the platform.
   
   **BINARY_CONTAINER_dpdpuv3_wrapper_OBJS:** optional, applied when add user xo
   
   **CLFLAGS_SLR:** optional, vitis command for user xo
   
   Then execute the commands below to generate xclbin files for DPU: 
   ```
   make gen_xclbin
   ```
   
   Outputs will be saved in the directory: projects/ml_shell/outputs
   
 **Please refer to PG400 for more detailed IP information**

