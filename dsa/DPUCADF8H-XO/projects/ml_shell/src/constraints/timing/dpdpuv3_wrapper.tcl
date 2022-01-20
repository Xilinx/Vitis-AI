
puts "Start to source [info script]"

global idx
set s_ram_cells [filter [all_fanout -from [get_nets [expr \$SLR$DPU_SLR_IDX($idx)_DPU_V3_WRAP_TOP]/u_axi_clock_converter_512/s_axi_aclk] -flat -endpoints_only -only_cells] {PRIMITIVE_SUBGROUP==dram || PRIMITIVE_SUBGROUP==LUTRAM}]
set m_ram_cells [filter [all_fanout -from [get_nets [expr \$SLR$DPU_SLR_IDX($idx)_DPU_V3_WRAP_TOP]/u_axi_clock_converter_512/m_axi_aclk] -flat -endpoints_only -only_cells] {PRIMITIVE_SUBGROUP==dram || PRIMITIVE_SUBGROUP==LUTRAM}]
set_false_path -from [get_pins -of $s_ram_cells -filter {REF_PIN_NAME == CLK}] -through [get_pins -of $s_ram_cells -filter {REF_PIN_NAME == O}] 
set_false_path -from [get_pins -of $m_ram_cells -filter {REF_PIN_NAME == CLK}] -through [get_pins -of $m_ram_cells -filter {REF_PIN_NAME == O}] 





set TOP \$SLR$DPU_SLR_IDX($idx)_DPU_V3_WRAP_TOP
create_generated_clock -name SLR$DPU_SLR_IDX($idx)_ACLK    [get_pins [expr $TOP]/BUFGCE_DIV_2/O] 
create_generated_clock -name SLR$DPU_SLR_IDX($idx)_ACLK_DR [get_pins [expr $TOP]/u_clk_wiz/inst/mmcme4_adv_inst/CLKOUT0]
set_property CLOCK_DELAY_GROUP GROUP_SLR$DPU_SLR_IDX($idx) [get_nets [expr $TOP]/u_clk_wiz/inst/clk_out_clk_wiz] [get_nets [expr $TOP]/dpdpuv3_top_data_aclk]
set_property CLOCK_DEDICATED_ROUTE FALSE       [get_nets [expr $TOP]/u_clk_wiz/inst/clk_out_clk_wiz]



puts "Start to source [info script]"

global DPU_V3_TOP

set all_regs_list [get_cells -hier -filter {PRIMITIVE_TYPE =~ REGISTER.SDR.*}]

set_multicycle_path 3 -setup -from [get_pins -of $all_regs_list -filter name=~$DPU_V3_TOP/u_misc_com/u_misc_ins_parser/ele_reg_wr_width_o_reg[*]/C]         
set_multicycle_path 2 -hold -from [get_pins -of $all_regs_list -filter name=~$DPU_V3_TOP/u_misc_com/u_misc_ins_parser/ele_reg_wr_width_o_reg[*]/C]         
set_multicycle_path 3 -setup -from [get_pins -of $all_regs_list -filter name=~$DPU_V3_TOP/u_misc_com/u_misc_ins_parser/ele_reg_wr_bank_id_o_reg[*]/C]         
set_multicycle_path 2 -hold -from [get_pins -of $all_regs_list -filter name=~$DPU_V3_TOP/u_misc_com/u_misc_ins_parser/ele_reg_wr_bank_id_o_reg[*]/C]         
set_multicycle_path 3 -setup -from [get_pins -of $all_regs_list -filter name=~$DPU_V3_TOP/u_misc_com/u_misc_ins_parser/ele_reg_wr_st_addr_o_reg[*]/C]         
set_multicycle_path 2 -hold -from [get_pins -of $all_regs_list -filter name=~$DPU_V3_TOP/u_misc_com/u_misc_ins_parser/ele_reg_wr_st_addr_o_reg[*]/C]         
set_multicycle_path 3 -setup -from [get_pins -of $all_regs_list -filter name=~$DPU_V3_TOP/u_misc_com/u_misc_ins_parser/ele_reg_wr_jump_o_reg[*]/C]            
set_multicycle_path 2 -hold -from [get_pins -of $all_regs_list -filter name=~$DPU_V3_TOP/u_misc_com/u_misc_ins_parser/ele_reg_wr_jump_o_reg[*]/C]            
set_multicycle_path 3 -setup -from [get_pins -of $all_regs_list -filter name=~$DPU_V3_TOP/u_misc_com/u_misc_ins_parser/ele_reg_wr_jump_endl_o_reg[*]/C]       
set_multicycle_path 2 -hold -from [get_pins -of $all_regs_list -filter name=~$DPU_V3_TOP/u_misc_com/u_misc_ins_parser/ele_reg_wr_jump_endl_o_reg[*]/C]       
set_multicycle_path 3 -setup -from [get_pins -of $all_regs_list -filter name=~$DPU_V3_TOP/u_misc_com/u_misc_ins_parser/ele_reg_calc_nonlinear_type_o_reg[*]/C]
set_multicycle_path 2 -hold -from [get_pins -of $all_regs_list -filter name=~$DPU_V3_TOP/u_misc_com/u_misc_ins_parser/ele_reg_calc_nonlinear_type_o_reg[*]/C]
set_multicycle_path 3 -setup -from [get_pins -of $all_regs_list -filter name=~$DPU_V3_TOP/u_misc_com/u_misc_ins_parser/ele_reg_wr_strd_out_o_reg[*]/C]        
set_multicycle_path 2 -hold -from [get_pins -of $all_regs_list -filter name=~$DPU_V3_TOP/u_misc_com/u_misc_ins_parser/ele_reg_wr_strd_out_o_reg[*]/C]        
set_multicycle_path 3 -setup -from [get_pins -of $all_regs_list -filter name=~$DPU_V3_TOP/u_misc_com/u_misc_ins_parser/ele_reg_wr_strd_off_set_o_reg[*]/C]    
set_multicycle_path 2 -hold -from [get_pins -of $all_regs_list -filter name=~$DPU_V3_TOP/u_misc_com/u_misc_ins_parser/ele_reg_wr_strd_off_set_o_reg[*]/C]    
set_multicycle_path 3 -setup -from [get_pins -of $all_regs_list -filter name=~$DPU_V3_TOP/u_misc_com/u_misc_ins_parser/ele_reg_calc_width_o_reg[*]/C]         
set_multicycle_path 2 -hold -from [get_pins -of $all_regs_list -filter name=~$DPU_V3_TOP/u_misc_com/u_misc_ins_parser/ele_reg_calc_width_o_reg[*]/C]         
set_multicycle_path 3 -setup -from [get_pins -of $all_regs_list -filter name=~$DPU_V3_TOP/u_misc_com/u_misc_ins_parser/ele_reg_calc_sets_num_o_reg[*]/C]      
set_multicycle_path 2 -hold -from [get_pins -of $all_regs_list -filter name=~$DPU_V3_TOP/u_misc_com/u_misc_ins_parser/ele_reg_calc_sets_num_o_reg[*]/C]      
set_multicycle_path 3 -setup -from [get_pins -of $all_regs_list -filter name=~$DPU_V3_TOP/u_misc_com/u_misc_ins_parser/ele_reg_calc_chn_grps_o_reg[*]/C]      
set_multicycle_path 2 -hold -from [get_pins -of $all_regs_list -filter name=~$DPU_V3_TOP/u_misc_com/u_misc_ins_parser/ele_reg_calc_chn_grps_o_reg[*]/C]      
set_multicycle_path 3 -setup -from [get_pins -of $all_regs_list -filter name=~$DPU_V3_TOP/u_misc_com/u_misc_ins_parser/ele_reg_calc_shf_rd0_o_reg[*]/C]       
set_multicycle_path 2 -hold -from [get_pins -of $all_regs_list -filter name=~$DPU_V3_TOP/u_misc_com/u_misc_ins_parser/ele_reg_calc_shf_rd0_o_reg[*]/C]       
set_multicycle_path 3 -setup -from [get_pins -of $all_regs_list -filter name=~$DPU_V3_TOP/u_misc_com/u_misc_ins_parser/ele_reg_calc_shf_rd1_o_reg[*]/C]       
set_multicycle_path 2 -hold -from [get_pins -of $all_regs_list -filter name=~$DPU_V3_TOP/u_misc_com/u_misc_ins_parser/ele_reg_calc_shf_rd1_o_reg[*]/C]       
set_multicycle_path 3 -setup -from [get_pins -of $all_regs_list -filter name=~$DPU_V3_TOP/u_misc_com/u_misc_ins_parser/ele_reg_calc_shf_rd2_o_reg[*]/C]       
set_multicycle_path 2 -hold -from [get_pins -of $all_regs_list -filter name=~$DPU_V3_TOP/u_misc_com/u_misc_ins_parser/ele_reg_calc_shf_rd2_o_reg[*]/C]       
set_multicycle_path 3 -setup -from [get_pins -of $all_regs_list -filter name=~$DPU_V3_TOP/u_misc_com/u_misc_ins_parser/ele_reg_calc_shf_rd3_o_reg[*]/C]       
set_multicycle_path 2 -hold -from [get_pins -of $all_regs_list -filter name=~$DPU_V3_TOP/u_misc_com/u_misc_ins_parser/ele_reg_calc_shf_rd3_o_reg[*]/C]       
set_multicycle_path 3 -setup -from [get_pins -of $all_regs_list -filter name=~$DPU_V3_TOP/u_misc_com/u_misc_ins_parser/pool_reg_wr_len_o_reg[*]/C]       
set_multicycle_path 2 -hold -from [get_pins -of $all_regs_list -filter name=~$DPU_V3_TOP/u_misc_com/u_misc_ins_parser/pool_reg_wr_len_o_reg[*]/C]       
set_multicycle_path 3 -setup -from [get_pins -of $all_regs_list -filter name=~$DPU_V3_TOP/u_misc_com/u_misc_ins_parser/pool_reg_wr_bid_out_o_reg[*]/C]       
set_multicycle_path 2 -hold -from [get_pins -of $all_regs_list -filter name=~$DPU_V3_TOP/u_misc_com/u_misc_ins_parser/pool_reg_wr_bid_out_o_reg[*]/C]       
set_multicycle_path 3 -setup -from [get_pins -of $all_regs_list -filter name=~$DPU_V3_TOP/u_misc_com/u_misc_ins_parser/pool_reg_wr_addr_out_o_reg[*]/C]      
set_multicycle_path 2 -hold -from [get_pins -of $all_regs_list -filter name=~$DPU_V3_TOP/u_misc_com/u_misc_ins_parser/pool_reg_wr_addr_out_o_reg[*]/C]      
set_multicycle_path 3 -setup -from [get_pins -of $all_regs_list -filter name=~$DPU_V3_TOP/u_misc_com/u_misc_ins_parser/pool_reg_wr_jump_o_reg[*]/C]          
set_multicycle_path 2 -hold -from [get_pins -of $all_regs_list -filter name=~$DPU_V3_TOP/u_misc_com/u_misc_ins_parser/pool_reg_wr_jump_o_reg[*]/C]          
set_multicycle_path 3 -setup -from [get_pins -of $all_regs_list -filter name=~$DPU_V3_TOP/u_misc_com/u_misc_ins_parser/pool_reg_wr_jump_endl_o_reg[*]/C]     
set_multicycle_path 2 -hold -from [get_pins -of $all_regs_list -filter name=~$DPU_V3_TOP/u_misc_com/u_misc_ins_parser/pool_reg_wr_jump_endl_o_reg[*]/C]     
set_multicycle_path 3 -setup -from [get_pins -of $all_regs_list -filter name=~$DPU_V3_TOP/u_misc_com/u_misc_ins_parser/pool_reg_calc_pmode_o_reg[*]/C]       
set_multicycle_path 2 -hold -from [get_pins -of $all_regs_list -filter name=~$DPU_V3_TOP/u_misc_com/u_misc_ins_parser/pool_reg_calc_pmode_o_reg[*]/C]       
set_multicycle_path 3 -setup -from [get_pins -of $all_regs_list -filter name=~$DPU_V3_TOP/u_misc_com/u_misc_ins_parser/pool_reg_pad_ker_h_size_o_reg[*]/C]   
set_multicycle_path 2 -hold -from [get_pins -of $all_regs_list -filter name=~$DPU_V3_TOP/u_misc_com/u_misc_ins_parser/pool_reg_pad_ker_h_size_o_reg[*]/C]   
set_multicycle_path 3 -setup -from [get_pins -of $all_regs_list -filter name=~$DPU_V3_TOP/u_misc_com/u_misc_ins_parser/pool_reg_pad_ker_w_size_o_reg[*]/C]   
set_multicycle_path 2 -hold -from [get_pins -of $all_regs_list -filter name=~$DPU_V3_TOP/u_misc_com/u_misc_ins_parser/pool_reg_pad_ker_w_size_o_reg[*]/C]   
set_multicycle_path 3 -setup -from [get_pins -of $all_regs_list -filter name=~$DPU_V3_TOP/u_misc_com/u_misc_ins_parser/pool_reg_pad_pstride_h_o_reg[*]/C]    
set_multicycle_path 2 -hold -from [get_pins -of $all_regs_list -filter name=~$DPU_V3_TOP/u_misc_com/u_misc_ins_parser/pool_reg_pad_pstride_h_o_reg[*]/C]    
set_multicycle_path 3 -setup -from [get_pins -of $all_regs_list -filter name=~$DPU_V3_TOP/u_misc_com/u_misc_ins_parser/pool_reg_pad_pstride_w_o_reg[*]/C]    
set_multicycle_path 2 -hold -from [get_pins -of $all_regs_list -filter name=~$DPU_V3_TOP/u_misc_com/u_misc_ins_parser/pool_reg_pad_pstride_w_o_reg[*]/C]    
set_multicycle_path 3 -setup -from [get_pins -of $all_regs_list -filter name=~$DPU_V3_TOP/u_misc_com/u_misc_ins_parser/pool_reg_pad_r_o_reg[*]/C]            
set_multicycle_path 2 -hold -from [get_pins -of $all_regs_list -filter name=~$DPU_V3_TOP/u_misc_com/u_misc_ins_parser/pool_reg_pad_r_o_reg[*]/C]            
set_multicycle_path 3 -setup -from [get_pins -of $all_regs_list -filter name=~$DPU_V3_TOP/u_misc_com/u_misc_ins_parser/pool_reg_pad_l_o_reg[*]/C]            
set_multicycle_path 2 -hold -from [get_pins -of $all_regs_list -filter name=~$DPU_V3_TOP/u_misc_com/u_misc_ins_parser/pool_reg_pad_l_o_reg[*]/C]            
set_multicycle_path 3 -setup -from [get_pins -of $all_regs_list -filter name=~$DPU_V3_TOP/u_misc_com/u_misc_ins_parser/pool_reg_pad_b_o_reg[*]/C]            
set_multicycle_path 2 -hold -from [get_pins -of $all_regs_list -filter name=~$DPU_V3_TOP/u_misc_com/u_misc_ins_parser/pool_reg_pad_b_o_reg[*]/C]            
set_multicycle_path 3 -setup -from [get_pins -of $all_regs_list -filter name=~$DPU_V3_TOP/u_misc_com/u_misc_ins_parser/pool_reg_pad_t_o_reg[*]/C]            
set_multicycle_path 2 -hold -from [get_pins -of $all_regs_list -filter name=~$DPU_V3_TOP/u_misc_com/u_misc_ins_parser/pool_reg_pad_t_o_reg[*]/C]            
set_multicycle_path 3 -setup -from [get_pins -of $all_regs_list -filter name=~$DPU_V3_TOP/u_misc_com/u_misc_ins_parser/pool_reg_pad_len_o_reg[*]/C]          
set_multicycle_path 2 -hold -from [get_pins -of $all_regs_list -filter name=~$DPU_V3_TOP/u_misc_com/u_misc_ins_parser/pool_reg_pad_len_o_reg[*]/C]          
set_multicycle_path 3 -setup -from [get_pins -of $all_regs_list -filter name=~$DPU_V3_TOP/u_misc_com/u_misc_ins_parser/pool_reg_pad_ch_grp_o_reg[*]/C]       
set_multicycle_path 2 -hold -from [get_pins -of $all_regs_list -filter name=~$DPU_V3_TOP/u_misc_com/u_misc_ins_parser/pool_reg_pad_ch_grp_o_reg[*]/C]       
set_multicycle_path 3 -setup -from [get_pins -of $all_regs_list -filter name=~$DPU_V3_TOP/u_misc_com/u_misc_ins_parser/pool_reg_pad_pmode_o_reg[*]/C]        
set_multicycle_path 2 -hold -from [get_pins -of $all_regs_list -filter name=~$DPU_V3_TOP/u_misc_com/u_misc_ins_parser/pool_reg_pad_pmode_o_reg[*]/C]        
puts "Start to source [info script]"

global DPU_V3_TOP



set_multicycle_path 4 -setup -start -from [get_pins -of [get_cells -hier -filter {PRIMITIVE_TYPE =~ REGISTER.SDR.*}] -filter name=~$DPU_V3_TOP/core_array[*].u_core/u_conv_top_core/u_conv_pad_core/cho[*].pe_bias_r_reg[*]/C] -to [get_pins -of [get_cells -hier -filter {PRIMITIVE_TYPE =~ REGISTER.SDR.*}] -filter name=~$DPU_V3_TOP/core_array[*].u_core/u_conv_top_core/u_conv_pad_core/pe_bias_rr_reg[*]/D]
set_multicycle_path 3 -hold -from [get_pins -of [get_cells -hier -filter {PRIMITIVE_TYPE =~ REGISTER.SDR.*}] -filter name=~$DPU_V3_TOP/core_array[*].u_core/u_conv_top_core/u_conv_pad_core/cho[*].pe_bias_r_reg[*]/C] -to [get_pins -of [get_cells -hier -filter {PRIMITIVE_TYPE =~ REGISTER.SDR.*}] -filter name=~$DPU_V3_TOP/core_array[*].u_core/u_conv_top_core/u_conv_pad_core/pe_bias_rr_reg[*]/D]











set_multicycle_path 4 -setup -start -from [get_pins -of [get_cells -hier -filter {PRIMITIVE_TYPE =~ REGISTER.SDR.*}] -filter name=~$DPU_V3_TOP/core_array[*].u_core/u_conv_top_core/u_fetch_res/data0_reg[*]/C] -through [get_pins -of [get_cells -hier -filter {PRIMITIVE_TYPE =~ REGISTER.SDR.*}] -filter name=~$DPU_V3_TOP/core_array[*].u_core/u_conv_top_core/u_fetch_res/data0_reg[*]/Q]
set_multicycle_path 3 -hold -from [get_pins -of [get_cells -hier -filter {PRIMITIVE_TYPE =~ REGISTER.SDR.*}] -filter name=~$DPU_V3_TOP/core_array[*].u_core/u_conv_top_core/u_fetch_res/data0_reg[*]/C] -through [get_pins -of [get_cells -hier -filter {PRIMITIVE_TYPE =~ REGISTER.SDR.*}] -filter name=~$DPU_V3_TOP/core_array[*].u_core/u_conv_top_core/u_fetch_res/data0_reg[*]/Q]

set_multicycle_path 4 -setup -start -from [get_pins -of [get_cells -hier -filter {PRIMITIVE_TYPE =~ REGISTER.SDR.*}] -filter name=~$DPU_V3_TOP/core_array[*].u_core/u_conv_top_core/u_fetch_res/data1_reg[*]/C] -through [get_pins -of [get_cells -hier -filter {PRIMITIVE_TYPE =~ REGISTER.SDR.*}] -filter name=~$DPU_V3_TOP/core_array[*].u_core/u_conv_top_core/u_fetch_res/data1_reg[*]/Q]
set_multicycle_path 3 -hold -from [get_pins -of [get_cells -hier -filter {PRIMITIVE_TYPE =~ REGISTER.SDR.*}] -filter name=~$DPU_V3_TOP/core_array[*].u_core/u_conv_top_core/u_fetch_res/data1_reg[*]/C] -through [get_pins -of [get_cells -hier -filter {PRIMITIVE_TYPE =~ REGISTER.SDR.*}] -filter name=~$DPU_V3_TOP/core_array[*].u_core/u_conv_top_core/u_fetch_res/data1_reg[*]/Q]
puts "Start to source [info script]"

global DPU_V3_TOP

set all_reg_list [get_cells -hier -filter {PRIMITIVE_TYPE =~ REGISTER.SDR.*}]

set_multicycle_path 2 -setup -to [get_pins -of $all_reg_list -filter {NAME =~ "*u_a_i_in_door_reg_en*/D"}]

set_multicycle_path 1 -hold -end -to [get_pins -of $all_reg_list -filter {NAME =~ "*u_a_i_in_door_reg_en*/D"}]

set_multicycle_path 2 -setup -to [get_pins -of $all_reg_list -filter {NAME =~ "*u_d_i_in_door_reg_en*/D"}]

set_multicycle_path 1 -hold -end -to [get_pins -of $all_reg_list -filter {NAME =~ "*u_d_i_in_door_reg_en*/D"}]


set_multicycle_path 2 -setup -to [get_pins -of $all_reg_list -filter "NAME =~ *u_regs_en_doutb_s0*/D"]
set_multicycle_path 1 -hold -end -to [get_pins -of $all_reg_list -filter "NAME =~ *u_regs_en_doutb_s0*/D"]










set_false_path -from    [get_pins -of [get_cells -hierarchical -filter name=~$DPU_V3_TOP/u_flowctrl/reg_*      ] -filter {IS_CLOCK == TRUE} ]
set_false_path -from    [get_pins -of [get_cells -hierarchical -filter name=~$DPU_V3_TOP/u_flowctrl/reg_*      ] -filter {IS_CLOCK == TRUE} ]

set_false_path -to      [get_pins -of [get_cells -hier -filter {PRIMITIVE_TYPE =~ REGISTER.SDR.*}] -filter name=~$DPU_V3_TOP/u_flowctrl/rdata*/D]

set_multicycle_path 2 -setup -from [get_pins -of [get_cells -hier -filter {PRIMITIVE_TYPE =~ REGISTER.SDR.*}] -filter name=~$DPU_V3_TOP/u_flowctrl/task_*/C]
set_multicycle_path 1 -hold -end -from [get_pins -of [get_cells -hier -filter {PRIMITIVE_TYPE =~ REGISTER.SDR.*}] -filter name=~$DPU_V3_TOP/u_flowctrl/task_*/C]

puts "Start to source [info script]"


set_false_path -from    [get_pins -of [get_cells -hier -filter {PRIMITIVE_TYPE =~ REGISTER.SDR.*}] -filter name=~$DPU_V3_TOP/u_flowctrl/reg_*/C]    -to     [get_pins -of [get_cells -hier -filter {PRIMITIVE_TYPE =~ REGISTER.SDR.*}] -filter name=~$DPU_V3_TOP/u_scheduler/u_profiler/*/R]

set_false_path -from    [get_pins -of [get_cells -hier -filter {PRIMITIVE_TYPE =~ REGISTER.SDR.*}] -filter name=~$DPU_V3_TOP/u_flowctrl/reg_*/C]    -to     [get_pins -of [get_cells -hier -filter {PRIMITIVE_TYPE =~ REGISTER.SDR.*}] -filter name=~$DPU_V3_TOP/u_scheduler/u_profiler/*/D]


set_false_path -from    [get_pins -of [get_cells -hier -filter {PRIMITIVE_TYPE =~ REGISTER.SDR.*}] -filter name=~$DPU_V3_TOP/u_flowctrl/reg_*/C]    -to     [get_pins -of [get_cells -hier -filter {PRIMITIVE_TYPE =~ REGISTER.SDR.*}] -filter name=~$DPU_V3_TOP/u_scheduler/u_profiler/*/CE]


set_false_path -from    [get_pins -of [get_cells -hier -filter {PRIMITIVE_TYPE =~ REGISTER.SDR.*}] -filter name=~$DPU_V3_TOP/u_flowctrl/reg_*/C]    -to     [get_pins -of [get_cells -hier -filter {PRIMITIVE_TYPE =~ REGISTER.SDR.*}] -filter name=~$DPU_V3_TOP/u_scheduler/u_fetch/*/D]


set_false_path -to      [get_pins -of [get_cells -hier -filter {PRIMITIVE_TYPE =~ REGISTER.SDR.*}] -filter name=~$DPU_V3_TOP/u_flowctrl/rdata*/D]   -from   [get_pins -of [get_cells -hier -filter {PRIMITIVE_TYPE =~ REGISTER.SDR.*}] -filter name=~$DPU_V3_TOP/u_scheduler/u_profiler/*/C]


set_false_path -from    [get_pins -of [get_cells -hier -filter {PRIMITIVE_TYPE =~ REGISTER.SDR.*}] -filter name=~$DPU_V3_TOP/u_flowctrl/reg_awcache_reg*/C]
set_false_path -from    [get_pins -of [get_cells -hier -filter {PRIMITIVE_TYPE =~ REGISTER.SDR.*}] -filter name=~$DPU_V3_TOP/u_flowctrl/reg_arcache_reg*/C]

set_multicycle_path 2 -setup -from [get_pins -of [get_cells -hier -filter {PRIMITIVE_TYPE =~ REGISTER.SDR.*}] -filter name=~$DPU_V3_TOP/u_flowctrl/task_*/C]    -to     [get_pins -of [get_cells -hier -filter {PRIMITIVE_TYPE =~ REGISTER.SDR.*}] -filter name=~$DPU_V3_TOP/u_scheduler/u_profiler/*/D]
set_multicycle_path 1 -hold -end -from [get_pins -of [get_cells -hier -filter {PRIMITIVE_TYPE =~ REGISTER.SDR.*}] -filter name=~$DPU_V3_TOP/u_flowctrl/task_*/C]    -to     [get_pins -of [get_cells -hier -filter {PRIMITIVE_TYPE =~ REGISTER.SDR.*}] -filter name=~$DPU_V3_TOP/u_scheduler/u_profiler/*/D]

set_multicycle_path 2 -setup -from [get_pins -of [get_cells -hier -filter {PRIMITIVE_TYPE =~ REGISTER.SDR.*}] -filter name=~$DPU_V3_TOP/u_flowctrl/task_*/C]    -to     [get_pins -of [get_cells -hier -filter {PRIMITIVE_TYPE =~ REGISTER.SDR.*}] -filter name=~$DPU_V3_TOP/u_scheduler/u_fetch/*/D]
set_multicycle_path 1 -hold -end -from [get_pins -of [get_cells -hier -filter {PRIMITIVE_TYPE =~ REGISTER.SDR.*}] -filter name=~$DPU_V3_TOP/u_flowctrl/task_*/C]    -to     [get_pins -of [get_cells -hier -filter {PRIMITIVE_TYPE =~ REGISTER.SDR.*}] -filter name=~$DPU_V3_TOP/u_scheduler/u_fetch/*/D]



set_false_path -to      [get_pins -of [get_cells -hier -filter {PRIMITIVE_TYPE =~ REGISTER.SDR.*}] -filter name=~$DPU_V3_TOP/u_save/u_ddr_writer/axvalid_reg/D]
set_false_path -to      [get_pins -of [get_cells -hier -filter {PRIMITIVE_TYPE =~ REGISTER.SDR.*}] -filter name=~$DPU_V3_TOP/u_save/u_ddr_writer/axaddr_reg[*]/D]
set_false_path -to      [get_pins -of [get_cells -hier -filter {PRIMITIVE_TYPE =~ REGISTER.SDR.*}] -filter name=~$DPU_V3_TOP/u_load/u_ddr_reader/axvalid_reg/D]
set_false_path -to      [get_pins -of [get_cells -hier -filter {PRIMITIVE_TYPE =~ REGISTER.SDR.*}] -filter name=~$DPU_V3_TOP/u_load/u_ddr_reader/axaddr_reg[*]/D]
set_false_path -to      [get_pins -of [get_cells -hier -filter {PRIMITIVE_TYPE =~ REGISTER.SDR.*}] -filter name=~$DPU_V3_TOP/u_save/u_ddr_writer/axlen_reg[*]/D]
set_false_path -to      [get_pins -of [get_cells -hier -filter {PRIMITIVE_TYPE =~ REGISTER.SDR.*}] -filter name=~$DPU_V3_TOP/u_save/u_ddr_writer/axlen_reg[*]/CE]
set_false_path -to      [get_pins -of [get_cells -hier -filter {PRIMITIVE_TYPE =~ REGISTER.SDR.*}] -filter name=~$DPU_V3_TOP/u_load/u_ddr_reader/axlen_reg[*]/D]
set_false_path -to      [get_pins -of [get_cells -hier -filter {PRIMITIVE_TYPE =~ REGISTER.SDR.*}] -filter name=~$DPU_V3_TOP/u_load/u_ddr_reader/axlen_reg[*]/CE]


