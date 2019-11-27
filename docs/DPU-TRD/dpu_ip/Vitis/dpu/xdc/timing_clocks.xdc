##########################
# constrains for DDR-DSP #
##########################

set clk_1x [get_clocks -of_objects [get_ports aclk]]
set clk_2x [get_clocks -of_objects [get_ports ap_clk_2]]

set_multicycle_path 2 -setup -start -from $clk_2x -to $clk_1x
set_multicycle_path 1 -hold  -start -from $clk_2x -to $clk_1x

set_multicycle_path 2 -setup -end   -from $clk_1x -to $clk_2x
set_multicycle_path 1 -hold  -end   -from $clk_1x -to $clk_2x

