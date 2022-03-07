hbm_memory_subsystem::force_host_port 28 1 [get_bd_cells hmss_0]
hbm_memory_subsystem::force_kernel_port [get_bd_intf_pins /dpu_0/DPU_AXI_0] 0 1 [get_bd_cells hmss_0]
hbm_memory_subsystem::force_kernel_port [get_bd_intf_pins /dpu_0/DPU_AXI_3] 1 1 [get_bd_cells hmss_0]
hbm_memory_subsystem::force_kernel_port [get_bd_intf_pins /dpu_1/DPU_AXI_0] 2 1 [get_bd_cells hmss_0]
hbm_memory_subsystem::force_kernel_port [get_bd_intf_pins /dpu_1/DPU_AXI_3] 3 1 [get_bd_cells hmss_0]
hbm_memory_subsystem::force_kernel_port [get_bd_intf_pins /dpu_1/DPU_AXI_W0] 4 1 [get_bd_cells hmss_0]
hbm_memory_subsystem::force_kernel_port [get_bd_intf_pins /dpu_1/DPU_AXI_W1] 5 1 [get_bd_cells hmss_0]
hbm_memory_subsystem::force_kernel_port [get_bd_intf_pins /dpu_1/DPU_AXI_I0] 6 1 [get_bd_cells hmss_0]
hbm_memory_subsystem::force_kernel_port [get_bd_intf_pins /dpu_0/DPU_AXI_1] 16 1 [get_bd_cells hmss_0]
hbm_memory_subsystem::force_kernel_port [get_bd_intf_pins /dpu_0/DPU_AXI_2] 17 1 [get_bd_cells hmss_0]
hbm_memory_subsystem::force_kernel_port [get_bd_intf_pins /dpu_1/DPU_AXI_1] 18 1 [get_bd_cells hmss_0]
hbm_memory_subsystem::force_kernel_port [get_bd_intf_pins /dpu_1/DPU_AXI_2] 19 1 [get_bd_cells hmss_0]
hbm_memory_subsystem::force_kernel_port [get_bd_intf_pins /dpu_0/DPU_AXI_W0] 20 1 [get_bd_cells hmss_0]
hbm_memory_subsystem::force_kernel_port [get_bd_intf_pins /dpu_0/DPU_AXI_W1] 21 1 [get_bd_cells hmss_0]
hbm_memory_subsystem::force_kernel_port [get_bd_intf_pins /dpu_0/DPU_AXI_I0] 22 1 [get_bd_cells hmss_0]




set __props [get_property CONFIG.ADVANCED_PROPERTIES [get_bd_cells hmss_0]]
# Put any other IP_OVERRIDE in here!!
dict set __props IP_OVERRIDE hbm_inst {CONFIG.USER_HBM_TCK_0 550 CONFIG.USER_HBM_TCK_1 550}
set_property -dict [list CONFIG.ADVANCED_PROPERTIES $__props] [get_bd_cells hmss_0]

set_param bd.hooks.addr.debug_scoped_use_ms_name true
validate_bd_design -force
