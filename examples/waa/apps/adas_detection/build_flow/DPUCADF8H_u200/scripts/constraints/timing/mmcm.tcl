puts "Start to source [info script]"

if { $SHELL_VER=="201803" } {
    if { $BOARD == "u200" } {
        set_property CLOCK_DEDICATED_ROUTE ANY_CMT_COLUMN [get_nets pfm_top_i/static_region/slr1/base_clocking/clkwiz_kernel/inst/CLK_CORE_DRP_I/clk_inst/clk_out1]
        set_property CLOCK_DEDICATED_ROUTE ANY_CMT_COLUMN [get_nets pfm_top_i/static_region/slr1/base_clocking/clkwiz_kernel2/inst/CLK_CORE_DRP_I/clk_inst/clk_out1]
    } 
    if { $BOARD == "u250" } {
        set_property CLOCK_DEDICATED_ROUTE ANY_CMT_COLUMN [get_nets pfm_top_i/static_region/slr0/base_clocking/clkwiz_kernel/inst/CLK_CORE_DRP_I/clk_inst/clk_out1]
        set_property CLOCK_DEDICATED_ROUTE ANY_CMT_COLUMN [get_nets pfm_top_i/static_region/slr0/base_clocking/clkwiz_kernel2/inst/CLK_CORE_DRP_I/clk_inst/clk_out1]
    } 
}

#if { ($SHELL_VER=="202002")||($SHELL_VER=="aws") } {


if { ($SHELL_VER=="202002")&&($BOARD=="u200") } {
    set_property CLOCK_DEDICATED_ROUTE ANY_CMT_COLUMN [get_nets level0_i/ulp/ss_ucs/inst/aclk_kernel_00_hierarchy/clock_throttling_aclk_kernel_00/U0/Clk_Out] 
    set_property CLOCK_DEDICATED_ROUTE ANY_CMT_COLUMN [get_nets level0_i/ulp/ss_ucs/inst/aclk_kernel_01_hierarchy/clock_throttling_aclk_kernel_01/U0/Clk_Out] 
}

if { ($SHELL_VER=="202002")&&($BOARD=="u250") } {
    set_property CLOCK_DEDICATED_ROUTE ANY_CMT_COLUMN [get_nets level0_i/level1/level1_i/ulp/ss_ucs/inst/aclk_kernel_00_hierarchy/clock_throttling_aclk_kernel_00/U0/Clk_Out] 
    set_property CLOCK_DEDICATED_ROUTE ANY_CMT_COLUMN [get_nets level0_i/level1/level1_i/ulp/ss_ucs/inst/aclk_kernel_01_hierarchy/clock_throttling_aclk_kernel_01/U0/Clk_Out] 
}

if { ($SHELL_VER=="aws") } {
    set_property CLOCK_DEDICATED_ROUTE ANY_CMT_COLUMN [get_nets WRAPPER_INST/SH/kernel_clks_i/clkwiz_kernel_clk1/inst/CLK_CORE_DRP_I/clk_inst/clk_out1]

}

if { ($SHELL_VER=="microsoft") } {
   set_property CLOCK_DEDICATED_ROUTE BACKBONE [get_nets level0_i/level1/level1_i/ulp/ss_ucs/inst/clock_throttling_kernel2/U0/Clk_Out]
}
