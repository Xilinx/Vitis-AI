########################################################
# User setup Tcl, used in SYNTH_DESIGN.TCL.PRE
#
puts "Start to source [info script]"


########################################################
#  Modify based on your design


##########################
# set "true" or "false" for SLR constraints
# constraints used in sdx_memory_subsystem
# $SDA_PATH/scripts/constraints/property/config_mss.tcl   --> by ChenCheng
set BUFFER_STRATEGY "true"


########################################################
# Not frequently used user setup options
#
if { $SHELL_VER=="201803" } {
    if { $DPU_NUM==1 } {
        set ML_SHELL_TOP                pfm_top_i/dynamic_region/DPUCADF8H_1/inst
        set SLR${SLR}_DPU_V3_WRAP_TOP   pfm_top_i/dynamic_region/DPUCADF8H_1/inst
        set SLR${SLR}_DPU_V3_TOP        pfm_top_i/dynamic_region/DPUCADF8H_1/inst/u_dpdpuv3_top
    } elseif { ($DPU_NUM==2 )&&($BOARD=="u200")} {
            set SLR0_DPU_V3_WRAP_TOP   pfm_top_i/dynamic_region/DPUCADF8H_1/inst
            set SLR0_DPU_V3_TOP        pfm_top_i/dynamic_region/DPUCADF8H_1/inst/u_dpdpuv3_top
            set SLR2_DPU_V3_WRAP_TOP   pfm_top_i/dynamic_region/DPUCADF8H_2/inst
            set SLR2_DPU_V3_TOP        pfm_top_i/dynamic_region/DPUCADF8H_2/inst/u_dpdpuv3_top
    } else {
        for { set idx 0}  {$idx < $DPU_NUM} {incr idx} { 
            set SLR${idx}_DPU_V3_WRAP_TOP   pfm_top_i/dynamic_region/DPUCADF8H_[expr $idx+1]/inst
            set SLR${idx}_DPU_V3_TOP        pfm_top_i/dynamic_region/DPUCADF8H_[expr $idx+1]/inst/u_dpdpuv3_top
        }
    }
} elseif { ($SHELL_VER=="202002")&&($BOARD=="u200") } {
    if { $DPU_NUM==1 } {
        set ML_SHELL_TOP                level0_i/ulp/DPUCADF8H_1/inst
        set SLR${SLR}_DPU_V3_WRAP_TOP   level0_i/ulp/DPUCADF8H_1/inst
        set SLR${SLR}_DPU_V3_TOP        level0_i/ulp/DPUCADF8H_1/inst/u_dpdpuv3_top
    } else {
         set SLR0_DPU_V3_WRAP_TOP   level0_i/ulp/DPUCADF8H_1/inst
         set SLR0_DPU_V3_TOP        level0_i/ulp/DPUCADF8H_1/inst/u_dpdpuv3_top
         set SLR2_DPU_V3_WRAP_TOP   level0_i/ulp/DPUCADF8H_2/inst
         set SLR2_DPU_V3_TOP        level0_i/ulp/DPUCADF8H_2/inst/u_dpdpuv3_top
         set SLR3_DPU_V3_WRAP_TOP   level0_i/ulp/DPUCADF8H_3/inst
         set SLR3_DPU_V3_TOP        level0_i/ulp/DPUCADF8H_3/inst/u_dpdpuv3_top
    } 
} elseif { (($SHELL_VER=="202002")||($SHELL_VER=="microsoft"))&&($BOARD=="u250") } {
    if { $DPU_NUM==1 } {
        set ML_SHELL_TOP                level0_i/level1/level1_i/ulp/DPUCADF8H_1/inst
        set SLR${SLR}_DPU_V3_WRAP_TOP   level0_i/level1/level1_i/ulp/DPUCADF8H_1/inst
        set SLR${SLR}_DPU_V3_TOP        level0_i/level1/level1_i/ulp/DPUCADF8H_1/inst/u_dpdpuv3_top
    } else {
         set SLR0_DPU_V3_WRAP_TOP   level0_i/level1/level1_i/ulp/DPUCADF8H_1/inst
         set SLR0_DPU_V3_TOP        level0_i/level1/level1_i/ulp/DPUCADF8H_1/inst/u_dpdpuv3_top
         set SLR2_DPU_V3_WRAP_TOP   level0_i/level1/level1_i/ulp/DPUCADF8H_2/inst
         set SLR2_DPU_V3_TOP        level0_i/level1/level1_i/ulp/DPUCADF8H_2/inst/u_dpdpuv3_top
         set SLR3_DPU_V3_WRAP_TOP   level0_i/level1/level1_i/ulp/DPUCADF8H_3/inst
         set SLR3_DPU_V3_TOP        level0_i/level1/level1_i/ulp/DPUCADF8H_3/inst/u_dpdpuv3_top
    } 
} elseif { ($SHELL_VER=="aws") } {
        set SLR${SLR}_DPU_V3_WRAP_TOP   WRAPPER_INST/CL/DPUCADF8H_1/inst
        set SLR${SLR}_DPU_V3_TOP        WRAPPER_INST/CL/DPUCADF8H_1/inst/u_dpdpuv3_top
}






set DESIGN_NAME  dpdpuv3_wrapper
set OUTPUT_DIR   $SDA_PATH/outputs
set REPORTS_DIR  $OUTPUT_DIR/reports
set RESULTS_DIR  $OUTPUT_DIR/checkpoints
set LOG_DIR      $OUTPUT_DIR/logs
#set DESIGN_CELLS $ML_SHELL_TOP/u_dpdpuv3_top
#set CORE0_CELLS  $ML_SHELL_TOP/u_dpdpuv3_top/core_array[0].u_core

