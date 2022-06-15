puts "Start to source [info script]"

global DPU_V3_TOP

set all_reg_list [get_cells -hier -filter {PRIMITIVE_TYPE =~ REGISTER.SDR.*}]

set_multicycle_path 2 -setup -to [get_pins -of $all_reg_list -filter {NAME =~ "*u_a_i_in_door_reg_en*/D"}]

set_multicycle_path 1 -hold -end -to [get_pins -of $all_reg_list -filter {NAME =~ "*u_a_i_in_door_reg_en*/D"}]

set_multicycle_path 2 -setup -to [get_pins -of $all_reg_list -filter {NAME =~ "*u_d_i_in_door_reg_en*/D"}]

set_multicycle_path 1 -hold -end -to [get_pins -of $all_reg_list -filter {NAME =~ "*u_d_i_in_door_reg_en*/D"}]


set_multicycle_path 2 -setup -to [get_pins -of $all_reg_list -filter "NAME =~ *u_regs_en_doutb_s0*/D"]
set_multicycle_path 1 -hold -end -to [get_pins -of $all_reg_list -filter "NAME =~ *u_regs_en_doutb_s0*/D"]


# dma ctl path
#set_false_path -to [get_pins -of $all_reg_list -filter name=~$DPU_V3_TOP/ctrl_idle_sync_reg[0]/D]
#set_false_path -to [get_pins -of $all_reg_list -filter name=~$DPU_V3_TOP/ctrl_done_sync_reg[0]/D]
#set_false_path -to [get_pins -of $all_reg_list -filter name=~$DPU_V3_TOP/ap_start_sync_reg[0]/D]

# set_false_path -from    [get_pins -of [get_cells -hierarchical -filter name=~$DPU_V3_TOP/u_flowctrl/reg_preproc_only*            ] -filter {IS_CLOCK == TRUE} ]
# set_false_path -from    [get_pins -of [get_cells -hierarchical -filter name=~$DPU_V3_TOP/u_flowctrl/reg_preproc_bypass*             ] -filter {IS_CLOCK == TRUE} ]
# set_false_path -from    [get_pins -of [get_cells -hierarchical -filter name=~$DPU_V3_TOP/u_flowctrl/reg_addr_raw_img*      ] -filter {IS_CLOCK == TRUE} ]
# set_false_path -from    [get_pins -of [get_cells -hierarchical -filter name=~$DPU_V3_TOP/u_flowctrl/reg_addr_preprocessed_img*  ] -filter {IS_CLOCK == TRUE} ]
# set_false_path -from    [get_pins -of [get_cells -hierarchical -filter name=~$DPU_V3_TOP/u_flowctrl/reg_img_height*          ] -filter {IS_CLOCK == TRUE} ]
# set_false_path -from    [get_pins -of [get_cells -hierarchical -filter name=~$DPU_V3_TOP/u_flowctrl/reg_img_pad_left*          ] -filter {IS_CLOCK == TRUE} ]
# set_false_path -from    [get_pins -of [get_cells -hierarchical -filter name=~$DPU_V3_TOP/u_flowctrl/reg_img_pad_right*          ] -filter {IS_CLOCK == TRUE} ]
# set_false_path -from    [get_pins -of [get_cells -hierarchical -filter name=~$DPU_V3_TOP/u_flowctrl/reg_img_width*      ] -filter {IS_CLOCK == TRUE} ]
# set_false_path -from    [get_pins -of [get_cells -hierarchical -filter name=~$DPU_V3_TOP/u_flowctrl/reg_stride_raw_img*      ] -filter {IS_CLOCK == TRUE} ]
# set_false_path -from    [get_pins -of [get_cells -hierarchical -filter name=~$DPU_V3_TOP/u_flowctrl/reg_stride_preprocessed_img*  ] -filter {IS_CLOCK == TRUE} ]
# set_false_path -from    [get_pins -of [get_cells -hierarchical -filter name=~$DPU_V3_TOP/u_flowctrl/reg_num_img*] -filter {IS_CLOCK == TRUE} ]
# set_false_path -from    [get_pins -of [get_cells -hierarchical -filter name=~$DPU_V3_TOP/u_flowctrl/reg_awcache*] -filter {IS_CLOCK == TRUE} ]
# set_false_path -from    [get_pins -of [get_cells -hierarchical -filter name=~$DPU_V3_TOP/u_flowctrl/reg_arcache*] -filter {IS_CLOCK == TRUE} ]
# set_false_path -from    [get_pins -of [get_cells -hierarchical -filter name=~$DPU_V3_TOP/u_flowctrl/reg_addr_params*      ] -filter {IS_CLOCK == TRUE} ]
# set_false_path -from    [get_pins -of [get_cells -hierarchical -filter name=~$DPU_V3_TOP/u_flowctrl/reg_addr_swap*      ] -filter {IS_CLOCK == TRUE} ]
# set_false_path -from    [get_pins -of [get_cells -hierarchical -filter name=~$DPU_V3_TOP/u_flowctrl/reg_addr_rslt*  ] -filter {IS_CLOCK == TRUE} ]
# set_false_path -from    [get_pins -of [get_cells -hierarchical -filter name=~$DPU_V3_TOP/u_flowctrl/reg_stride_rslt*  ] -filter {IS_CLOCK == TRUE} ]
# set_false_path -from    [get_pins -of [get_cells -hierarchical -filter name=~$DPU_V3_TOP/u_flowctrl/reg_img_pad_value*      ] -filter {IS_CLOCK == TRUE} ]
# set_false_path -from    [get_pins -of [get_cells -hierarchical -filter name=~$DPU_V3_TOP/u_flowctrl/reg_img_mean_value*     ] -filter {IS_CLOCK == TRUE} ]
# set_false_path -from    [get_pins -of [get_cells -hierarchical -filter name=~$DPU_V3_TOP/u_flowctrl/reg_img_stride*     ] -filter {IS_CLOCK == TRUE} ]
# set_false_path -from    [get_pins -of [get_cells -hierarchical -filter name=~$DPU_V3_TOP/u_flowctrl/reg_img_krnl_size*      ] -filter {IS_CLOCK == TRUE} ]
# set_false_path -from    [get_pins -of [get_cells -hierarchical -filter name=~$DPU_V3_TOP/u_flowctrl/reg_img_factor*      ] -filter {IS_CLOCK == TRUE} ]

#set_false_path -from    [get_pins -of [get_cells -hierarchical -filter name=~$DPU_V3_TOP/u_flowctrl/reg_*      ] -filter {IS_CLOCK == TRUE} ]
#set_false_path -from    [get_pins -of [get_cells -hierarchical -filter name=~$DPU_V3_TOP/u_flowctrl/reg_*      ] -filter {IS_CLOCK == TRUE} ]





# for new task queue

# false path
set_false_path -from    [get_pins -of [get_cells -hierarchical -filter name=~$DPU_V3_TOP/u_flowctrl/reg_*      ] -filter {IS_CLOCK == TRUE} ]
set_false_path -from    [get_pins -of [get_cells -hierarchical -filter name=~$DPU_V3_TOP/u_flowctrl/reg_*      ] -filter {IS_CLOCK == TRUE} ]

set_false_path -to      [get_pins -of [get_cells -hier -filter {PRIMITIVE_TYPE =~ REGISTER.SDR.*}] -filter name=~$DPU_V3_TOP/u_flowctrl/rdata*/D]

#mcp 
#profiler
set_multicycle_path 2 -setup -from [get_pins -of [get_cells -hier -filter {PRIMITIVE_TYPE =~ REGISTER.SDR.*}] -filter name=~$DPU_V3_TOP/u_flowctrl/task_*/C]
set_multicycle_path 1 -hold -end -from [get_pins -of [get_cells -hier -filter {PRIMITIVE_TYPE =~ REGISTER.SDR.*}] -filter name=~$DPU_V3_TOP/u_flowctrl/task_*/C]

