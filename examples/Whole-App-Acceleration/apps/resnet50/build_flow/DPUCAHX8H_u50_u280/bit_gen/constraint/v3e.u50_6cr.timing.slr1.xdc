create_generated_clock -name ACLK1 [get_pins "level0_i/ulp/dpu_1/inst/v3e_bd_i/dpu_top_0/inst/dpu_clock_gen_inst/u_dpu_mmcm/inst/mmcme4_adv_inst/CLKOUT1"]
create_generated_clock -name ACLK_DR1 [get_pins "level0_i/ulp/dpu_1/inst/v3e_bd_i/dpu_top_0/inst/dpu_clock_gen_inst/u_dpu_mmcm/inst/mmcme4_adv_inst/CLKOUT0"]
create_generated_clock -name ACLK_REG1 [get_pins "level0_i/ulp/dpu_1/inst/v3e_bd_i/dpu_top_0/inst/dpu_clock_gen_inst/u_dpu_mmcm/inst/mmcme4_adv_inst/CLKOUT2"]

#set_multicycle_path -setup 2 -from [get_pins " \
#level0_i/ulp/dpu_1/inst/v3e_bd_i/dpu_clock_gen_0/inst/RESET_REG_CONTROL[3].reset_reg_counter_reg[3]*/C\
#"]
#set_multicycle_path -hold  1 -from [get_pins " \
#level0_i/ulp/dpu_1/inst/v3e_bd_i/dpu_clock_gen_0/inst/RESET_REG_CONTROL[3].reset_reg_counter_reg[3]*/C\
#"]

#set_property CLOCK_DELAY_GROUP CGRP_SLR1 [get_nets "level0_i/ulp/dpu_1/inst/v3e_bd_i/dpu_top_0/inst/ACLK_B level0_i/ulp/dpu_1/inst/v3e_bd_i/dpu_top_0/inst/ACLK_C level0_i/ulp/dpu_1/inst/v3e_bd_i/dpu_top_0/inst/ACLK_C_DR level0_i/ulp/dpu_1/inst/v3e_bd_i/dpu_top_0/inst/ACLK_F level0_i/ulp/dpu_1/inst/v3e_bd_i/dpu_top_0/inst/ACLK_LI level0_i/ulp/dpu_1/inst/v3e_bd_i/dpu_top_0/inst/ACLK_LW level0_i/ulp/dpu_1/inst/v3e_bd_i/dpu_top_0/inst/ACLK_M level0_i/ulp/dpu_1/inst/v3e_bd_i/dpu_top_0/inst/ACLK_S level0_i/ulp/dpu_1/inst/v3e_bd_i/dpu_top_0/inst/ACLK_SW"]
set_property CLOCK_DELAY_GROUP CGRP_SLR1 [get_nets "level0_i/ulp/dpu_1/inst/v3e_bd_i/dpu_top_0/inst/ACLK_B level0_i/ulp/dpu_1/inst/v3e_bd_i/dpu_top_0/inst/ACLK_C level0_i/ulp/dpu_1/inst/v3e_bd_i/dpu_top_0/inst/ACLK_C_DR level0_i/ulp/dpu_1/inst/v3e_bd_i/dpu_top_0/inst/ACLK_CS level0_i/ulp/dpu_1/inst/v3e_bd_i/dpu_top_0/inst/ACLK_OUT level0_i/ulp/dpu_1/inst/v3e_bd_i/dpu_top_0/inst/ACLK_LI level0_i/ulp/dpu_1/inst/v3e_bd_i/dpu_top_0/inst/ACLK_LW level0_i/ulp/dpu_1/inst/v3e_bd_i/dpu_top_0/inst/ACLK_M level0_i/ulp/dpu_1/inst/v3e_bd_i/dpu_top_0/inst/ACLK_S level0_i/ulp/dpu_1/inst/v3e_bd_i/dpu_top_0/inst/ACLK_SW"]
#set_property CLOCK_DELAY_GROUP CGRP_SLR1 [get_nets "level0_i/ulp/dpu_1/inst/v3e_bd_i/dpu_top_0/inst/ACLK_C level0_i/ulp/dpu_1/inst/v3e_bd_i/dpu_top_0/inst/ACLK_C_DR level0_i/ulp/dpu_1/inst/v3e_bd_i/dpu_top_0/inst/ACLK_CS level0_i/ulp/dpu_1/inst/v3e_bd_i/dpu_top_0/inst/ACLK_OUT"]
#set_property CLOCK_DELAY_GROUP CGRP_SLR1_ROOT [get_nets "level0_i/ulp/dpu_1/inst/v3e_bd_i/dpu_top_0/inst/ACLK level0_i/ulp/dpu_1/inst/v3e_bd_i/dpu_top_0/inst/ACLK_DR"]
