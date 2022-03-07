puts "Start to source [info script]"




set BUFFER_STRATEGY "true"


if { ($PLATFORM=="xilinx_u200_xdma_201830_2")||($PLATFORM=="xilinx_u250_xdma_201830_2") } {
    for { set idx 0}  {$idx < $DPU_NUM} {incr idx} { 
    set SLR$DPU_SLR_IDX($idx)_DPU_V3_TOP        pfm_top_i/dynamic_region/DPUCADF8H_[expr $idx+1]/inst/u_dpdpuv3_top
    set SLR$DPU_SLR_IDX($idx)_DPU_V3_WRAP_TOP   pfm_top_i/dynamic_region/DPUCADF8H_[expr $idx+1]/inst
    }
}

if { ($PLATFORM=="xilinx_u250_gen3x16_xdma_2_1") ||($PLATFORM=="xilinx_u250_gen3x16_xdma_3_1")} {
    for { set idx 0}  {$idx < $DPU_NUM} {incr idx} { 
    set SLR$DPU_SLR_IDX($idx)_DPU_V3_TOP        level0_i/level1/level1_i/ulp/DPUCADF8H_[expr $idx+1]/inst/u_dpdpuv3_top
    set SLR$DPU_SLR_IDX($idx)_DPU_V3_WRAP_TOP   level0_i/level1/level1_i/ulp/DPUCADF8H_[expr $idx+1]/inst
    }
}

if { ($PLATFORM=="xilinx_u200_gen3x16_xdma_base_1") } {
    for { set idx 0}  {$idx < $DPU_NUM} {incr idx} { 
    set SLR$DPU_SLR_IDX($idx)_DPU_V3_TOP        level0_i/ulp/DPUCADF8H_[expr $idx+1]/inst/u_dpdpuv3_top
    set SLR$DPU_SLR_IDX($idx)_DPU_V3_WRAP_TOP   level0_i/ulp/DPUCADF8H_[expr $idx+1]/inst
    }
}

if { ($PLATFORM=="aws") } {
    for { set idx 0}  {$idx < $DPU_NUM} {incr idx} { 
    set SLR$DPU_SLR_IDX($idx)_DPU_V3_TOP        WRAPPER_INST/CL/DPUCADF8H_[expr $idx+1]/inst/u_dpdpuv3_top
    set SLR$DPU_SLR_IDX($idx)_DPU_V3_WRAP_TOP   WRAPPER_INST/CL/DPUCADF8H_[expr $idx+1]/inst
    }
}






set DESIGN_NAME  dpdpuv3_wrapper
set OUTPUT_DIR   $SDA_PATH/outputs
set REPORTS_DIR  $OUTPUT_DIR/reports
set RESULTS_DIR  $OUTPUT_DIR/checkpoints
set LOG_DIR      $OUTPUT_DIR/logs


