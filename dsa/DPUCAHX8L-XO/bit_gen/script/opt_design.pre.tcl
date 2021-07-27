#For u50
#set_property CLOCK_DEDICATED_ROUTE ANY_CMT_COLUMN [get_nets -hier -regexp .*clkwiz_kernel2/inst/CLK_CORE_DRP_I/clk_inst/clk_out1] 
#For u280
set_property CLOCK_DEDICATED_ROUTE ANY_CMT_COLUMN [get_nets pfm_top_i/static_region/base_clocking/clkwiz_kernel*/inst/CLK_CORE_DRP_I/clk_inst/clk_out1]

#Add physical constraints for better timing if needed
