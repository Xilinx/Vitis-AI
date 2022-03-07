set_property CLOCK_DEDICATED_ROUTE ANY_CMT_COLUMN [get_nets level0_i/ulp/ulp_ucs/inst/clkwiz_kernel2/inst/CLK_CORE_DRP_I/clk_inst/clk_out1]
#set_property CLOCK_DEDICATED_ROUTE BACKBONE [get_nets level0_i/ulp/ulp_ucs/inst/shell_utils_clock_throttling_kernel2/U0/Clk_Out]
#using 19.2 daily latest hier update to the following 
set_property CLOCK_DEDICATED_ROUTE BACKBONE [get_nets level0_i/ulp/ulp_ucs/inst/clock_throttling_kernel2/U0/Clk_Out]
#We do not need PR for SLR0 and SLR1
set_property SNAPPING_MODE NESTED [get_pblocks pblock_dynamic_SLR0]
set_property SNAPPING_MODE NESTED [get_pblocks pblock_dynamic_SLR1]

#remove clock root
#set_property USER_CLOCK_ROOT X3Y1 [get_nets level0_i/ulp/ulp_ucs/inst/shell_utils_clock_throttling_kernel/U0/Clk_Out]
#set_property USER_CLOCK_ROOT X3Y1 [get_nets level0_i/ulp/ulp_ucs/inst/shell_utils_clock_throttling_kernel2/U0/Clk_Out]

