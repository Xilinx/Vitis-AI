puts "Start to source [info script]"

source $SDA_PATH/src/constraints/property/vivado_property.tcl




 ##################
for { set idx 0}  {$idx < $DPU_NUM} {incr idx} {
    set DPU_V3_TOP [expr \$SLR$DPU_SLR_IDX($idx)_DPU_V3_TOP]
    source $SDA_PATH/src/constraints/timing/dpdpuv3_wrapper.tcl
}

if { $BOARD == "u200" } {
    for { set idx 0}  {$idx < $DPU_NUM} {incr idx} {
        group_path -name  SLR$DPU_SLR_IDX($idx)_ACLK_DR -weight 2
    }
}


for { set idx 0}  {$idx < $DPU_NUM} {incr idx} {
    set DPU_V3_TOP [expr \$SLR$DPU_SLR_IDX($idx)_DPU_V3_TOP]
    source $SDA_PATH/src/constraints/physical/${PLATFORM}_slr$DPU_SLR_IDX($idx)_physical.tcl
}



if { $PLATFORM == "xilinx_u200_xdma_201830_2" } {
    set_property CLOCK_DEDICATED_ROUTE ANY_CMT_COLUMN [get_nets pfm_top_i/static_region/slr1/base_clocking/clkwiz_kernel/inst/CLK_CORE_DRP_I/clk_inst/clk_out1]
    set_property CLOCK_DEDICATED_ROUTE ANY_CMT_COLUMN [get_nets pfm_top_i/static_region/slr1/base_clocking/clkwiz_kernel2/inst/CLK_CORE_DRP_I/clk_inst/clk_out1]
} 
if { $PLATFORM == "xilinx_u250_xdma_201830_2" } {
    set_property CLOCK_DEDICATED_ROUTE ANY_CMT_COLUMN [get_nets pfm_top_i/static_region/slr0/base_clocking/clkwiz_kernel/inst/CLK_CORE_DRP_I/clk_inst/clk_out1]
    set_property CLOCK_DEDICATED_ROUTE ANY_CMT_COLUMN [get_nets pfm_top_i/static_region/slr0/base_clocking/clkwiz_kernel2/inst/CLK_CORE_DRP_I/clk_inst/clk_out1]
} 


if { $PLATFORM=="xilinx_u200_gen3x16_xdma_base_1" } {
    set_property CLOCK_DEDICATED_ROUTE ANY_CMT_COLUMN [get_nets level0_i/ulp/ss_ucs/inst/aclk_kernel_00_hierarchy/clock_throttling_aclk_kernel_00/U0/Clk_Out] 
    set_property CLOCK_DEDICATED_ROUTE ANY_CMT_COLUMN [get_nets level0_i/ulp/ss_ucs/inst/aclk_kernel_01_hierarchy/clock_throttling_aclk_kernel_01/U0/Clk_Out] 
}

if { $PLATFORM=="xilinx_u250_gen3x16_xdma_3_1" } {
    set_property CLOCK_DEDICATED_ROUTE ANY_CMT_COLUMN [get_nets level0_i/level1/level1_i/ulp/ss_ucs/inst/aclk_kernel_00_hierarchy/clock_throttling_aclk_kernel_00/U0/Clk_Out] 
    set_property CLOCK_DEDICATED_ROUTE ANY_CMT_COLUMN [get_nets level0_i/level1/level1_i/ulp/ss_ucs/inst/aclk_kernel_01_hierarchy/clock_throttling_aclk_kernel_01/U0/Clk_Out] 
}

if { ($PLATFORM=="aws") } {
    set_property CLOCK_DEDICATED_ROUTE ANY_CMT_COLUMN [get_nets WRAPPER_INST/SH/kernel_clks_i/clkwiz_kernel_clk1/inst/CLK_CORE_DRP_I/clk_inst/clk_out1]
}

if { ($PLATFORM=="xilinx_u250_gen3x16_xdma_2_1") } {
   set_property CLOCK_DEDICATED_ROUTE BACKBONE [get_nets level0_i/level1/level1_i/ulp/ss_ucs/inst/clock_throttling_kernel2/U0/Clk_Out]
}




if { $BOARD == "u250" } {
    set_clock_uncertainty -setup 0.1 [get_clocks SLR*_ACLK_DR]
    set_clock_uncertainty -setup 0.2 [get_clocks SLR*_ACLK]
}




make_outputs "opt_design"
