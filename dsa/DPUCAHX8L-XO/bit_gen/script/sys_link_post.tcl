hbm_memory_subsystem::force_host_port 28 1 [get_bd_cells hmss_0]
hbm_memory_subsystem::force_kernel_port [get_bd_intf_pins /DPUCAHX8L_A/DPU_VB_M_AXI_00]  0 1  [get_bd_cells hmss_0]
hbm_memory_subsystem::force_kernel_port [get_bd_intf_pins /DPUCAHX8L_A/DPU_VB_M_AXI_01]  2 1  [get_bd_cells hmss_0]
hbm_memory_subsystem::force_kernel_port [get_bd_intf_pins /DPUCAHX8L_A/DPU_VB_M_AXI_02]  1 1  [get_bd_cells hmss_0]
hbm_memory_subsystem::force_kernel_port [get_bd_intf_pins /DPUCAHX8L_A/DPU_VB_M_AXI_03]  3 1  [get_bd_cells hmss_0]
hbm_memory_subsystem::force_kernel_port [get_bd_intf_pins /DPUCAHX8L_A/DPU_VB_M_AXI_04]  4 1  [get_bd_cells hmss_0]
hbm_memory_subsystem::force_kernel_port [get_bd_intf_pins /DPUCAHX8L_A/DPU_VB_M_AXI_05]  7 1  [get_bd_cells hmss_0]
hbm_memory_subsystem::force_kernel_port [get_bd_intf_pins /DPUCAHX8L_A/DPU_VB_M_AXI_06]  5 1  [get_bd_cells hmss_0]
hbm_memory_subsystem::force_kernel_port [get_bd_intf_pins /DPUCAHX8L_A/DPU_VB_M_AXI_07]  6 1  [get_bd_cells hmss_0]
hbm_memory_subsystem::force_kernel_port [get_bd_intf_pins /DPUCAHX8L_A/DPU_SYS_M_AXI_00]  26 1  [get_bd_cells hmss_0]
hbm_memory_subsystem::force_kernel_port [get_bd_intf_pins /DPUCAHX8L_A/DPU_SYS_M_AXI_01]  24 1  [get_bd_cells hmss_0]
hbm_memory_subsystem::force_kernel_port [get_bd_intf_pins /DPUCAHX8L_A/DPU_SYS_M_AXI_02]  25 1  [get_bd_cells hmss_0]
#Uncomment below line if you want to deploy 2 cores
#hbm_memory_subsystem::force_kernel_port [get_bd_intf_pins /DPUCAHX8L_B/DPU_VB_M_AXI_00]  8 1  [get_bd_cells hmss_0]
#hbm_memory_subsystem::force_kernel_port [get_bd_intf_pins /DPUCAHX8L_B/DPU_VB_M_AXI_01]  10 1 [get_bd_cells hmss_0]
#hbm_memory_subsystem::force_kernel_port [get_bd_intf_pins /DPUCAHX8L_B/DPU_VB_M_AXI_02]  9 1  [get_bd_cells hmss_0]
#hbm_memory_subsystem::force_kernel_port [get_bd_intf_pins /DPUCAHX8L_B/DPU_VB_M_AXI_03]  11 1  [get_bd_cells hmss_0]
#hbm_memory_subsystem::force_kernel_port [get_bd_intf_pins /DPUCAHX8L_B/DPU_VB_M_AXI_04]  12 1  [get_bd_cells hmss_0]
#hbm_memory_subsystem::force_kernel_port [get_bd_intf_pins /DPUCAHX8L_B/DPU_VB_M_AXI_05]  15 1  [get_bd_cells hmss_0]
#hbm_memory_subsystem::force_kernel_port [get_bd_intf_pins /DPUCAHX8L_B/DPU_VB_M_AXI_06]  13 1  [get_bd_cells hmss_0]
#hbm_memory_subsystem::force_kernel_port [get_bd_intf_pins /DPUCAHX8L_B/DPU_VB_M_AXI_07]  14 1  [get_bd_cells hmss_0]
#hbm_memory_subsystem::force_kernel_port [get_bd_intf_pins /DPUCAHX8L_B/DPU_SYS_M_AXI_00]  27 1  [get_bd_cells hmss_0]
#hbm_memory_subsystem::force_kernel_port [get_bd_intf_pins /DPUCAHX8L_B/DPU_SYS_M_AXI_01]  22 1  [get_bd_cells hmss_0]
#hbm_memory_subsystem::force_kernel_port [get_bd_intf_pins /DPUCAHX8L_B/DPU_SYS_M_AXI_02]  23 1  [get_bd_cells hmss_0]

set ap [get_property CONFIG.ADVANCED_PROPERTIES [get_bd_cells /hmss_0]]
dict set ap minimal_initial_configuration true                                                                                                                                                                                           set_property CONFIG.ADVANCED_PROPERTIES $ap [get_bd_cells /hmss_0]

set_param bd.hooks.addr.debug_scoped_use_ms_name true
assign_bd_address [get_bd_addr_segs {DPUCAHX8L_A/s_axi_control/reg0 }]
#Uncomment below line if you want to deploy 2 cores
#assign_bd_address [get_bd_addr_segs {DPUCAHX8L_B/s_axi_control/reg0 }]
validate_bd_design -force

