create_generated_clock -name ACLK0 [get_pins "pfm_top_i/dynamic_region/dpu_0/inst/v3e_bd_i/dpu_top_0/inst/dpu_clock_gen_inst/u_dpu_mmcm/inst/mmcme4_adv_inst/CLKOUT1"]
create_generated_clock -name ACLK_DR0 [get_pins "pfm_top_i/dynamic_region/dpu_0/inst/v3e_bd_i/dpu_top_0/inst/dpu_clock_gen_inst/u_dpu_mmcm/inst/mmcme4_adv_inst/CLKOUT0"]
create_generated_clock -name ACLK_REG0 [get_pins "pfm_top_i/dynamic_region/dpu_0/inst/v3e_bd_i/dpu_top_0/inst/dpu_clock_gen_inst/u_dpu_mmcm/inst/mmcme4_adv_inst/CLKOUT2"]
#group_path -name ACLK_DR -weight 2
#set_property CLOCK_DELAY_GROUP DPU_CLK_CLKGROUP "[get_nets -of [get_pins " \ 
#pfm_top_i/dynamic_region/dpu_0/inst/v3e_bd_i/bufgce_div_0/inst/BUFGCE_DIV_1/O \
#pfm_top_i/dynamic_region/dpu_0/inst/v3e_bd_i/bufgce_div_0/inst/BUFGCE_DIV_2/O \
#"]]"
#
#set_property USER_CLOCK_ROOT [get_clock_regions X3Y5] "[get_nets -of [get_pins " \ 
#pfm_top_i/dynamic_region/dpu_0/inst/v3e_bd_i/bufgce_div_0/inst/BUFGCE_DIV_1/O \
#pfm_top_i/dynamic_region/dpu_0/inst/v3e_bd_i/bufgce_div_0/inst/BUFGCE_DIV_2/O \
#"]]"
#
#set_property CLOCK_REGION [get_clock_regions X4Y6] "[get_cells "pfm_top_i/dynamic_region/dpu_0/inst/v3e_bd_i/bufgce_div_0/inst/BUFGCE_DIV_1"]"
#set_property CLOCK_REGION [get_clock_regions X4Y6] "[get_cells "pfm_top_i/dynamic_region/dpu_0/inst/v3e_bd_i/bufgce_div_0/inst/BUFGCE_DIV_2"]"


#set_property CLOCK_DELAY_GROUP CGRP_SLR0 [get_nets "pfm_top_i/dynamic_region/dpu_0/inst/v3e_bd_i/dpu_top_0/inst/ACLK_B pfm_top_i/dynamic_region/dpu_0/inst/v3e_bd_i/dpu_top_0/inst/ACLK_C pfm_top_i/dynamic_region/dpu_0/inst/v3e_bd_i/dpu_top_0/inst/ACLK_C_DR pfm_top_i/dynamic_region/dpu_0/inst/v3e_bd_i/dpu_top_0/inst/ACLK_F pfm_top_i/dynamic_region/dpu_0/inst/v3e_bd_i/dpu_top_0/inst/ACLK_LI pfm_top_i/dynamic_region/dpu_0/inst/v3e_bd_i/dpu_top_0/inst/ACLK_LW pfm_top_i/dynamic_region/dpu_0/inst/v3e_bd_i/dpu_top_0/inst/ACLK_M pfm_top_i/dynamic_region/dpu_0/inst/v3e_bd_i/dpu_top_0/inst/ACLK_S pfm_top_i/dynamic_region/dpu_0/inst/v3e_bd_i/dpu_top_0/inst/ACLK_SW"]
#set_property CLOCK_DELAY_GROUP CGRP_SLR0 [get_nets "pfm_top_i/dynamic_region/dpu_0/inst/v3e_bd_i/dpu_top_0/inst/ACLK_B pfm_top_i/dynamic_region/dpu_0/inst/v3e_bd_i/dpu_top_0/inst/ACLK_C pfm_top_i/dynamic_region/dpu_0/inst/v3e_bd_i/dpu_top_0/inst/ACLK_C_DR pfm_top_i/dynamic_region/dpu_0/inst/v3e_bd_i/dpu_top_0/inst/ACLK_CS pfm_top_i/dynamic_region/dpu_0/inst/v3e_bd_i/dpu_top_0/inst/ACLK_OUT pfm_top_i/dynamic_region/dpu_0/inst/v3e_bd_i/dpu_top_0/inst/ACLK_LI pfm_top_i/dynamic_region/dpu_0/inst/v3e_bd_i/dpu_top_0/inst/ACLK_LW pfm_top_i/dynamic_region/dpu_0/inst/v3e_bd_i/dpu_top_0/inst/ACLK_M pfm_top_i/dynamic_region/dpu_0/inst/v3e_bd_i/dpu_top_0/inst/ACLK_S pfm_top_i/dynamic_region/dpu_0/inst/v3e_bd_i/dpu_top_0/inst/ACLK_SW"]
