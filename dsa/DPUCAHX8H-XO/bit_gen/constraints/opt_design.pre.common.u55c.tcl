#set_property CLOCK_DEDICATED_ROUTE ANY_CMT_COLUMN [get_nets pfm_top_i/static_region/base_clocking/clkwiz_kernel*/inst/CLK_CORE_DRP_I/clk_inst/clk_out1]
set_property CLOCK_DEDICATED_ROUTE ANY_CMT_COLUMN [get_nets level0_i/ulp/ulp_ucs/inst/aclk_kernel_0*_hierarchy/clkwiz_aclk_kernel_01/inst/CLK_CORE_DRP_I/clk_inst/clk_out1]
set_property CLOCK_DEDICATED_ROUTE BACKBONE [get_nets level0_i/ulp/ulp_ucs/inst/aclk_kernel_*_hierarchy/clock_throttling_aclk_kernel_*/U*/Clk_Out]



