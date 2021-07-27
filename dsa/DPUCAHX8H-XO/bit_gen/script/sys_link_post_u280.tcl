hbm_memory_subsystem::force_host_port 8 1 [get_bd_cells hmss_0]
hbm_memory_subsystem::force_kernel_port [get_bd_intf_pins /dpu_0/DPU_AXI_0] 0 1 [get_bd_cells hmss_0]
hbm_memory_subsystem::force_kernel_port [get_bd_intf_pins /dpu_0/DPU_AXI_3] 1 1 [get_bd_cells hmss_0]
hbm_memory_subsystem::force_kernel_port [get_bd_intf_pins /dpu_1/DPU_AXI_0] 2 1 [get_bd_cells hmss_0]
hbm_memory_subsystem::force_kernel_port [get_bd_intf_pins /dpu_1/DPU_AXI_1] 3 1 [get_bd_cells hmss_0]
hbm_memory_subsystem::force_kernel_port [get_bd_intf_pins /dpu_1/DPU_AXI_4] 4 1 [get_bd_cells hmss_0]
hbm_memory_subsystem::force_kernel_port [get_bd_intf_pins /dpu_2/DPU_AXI_0] 5 1 [get_bd_cells hmss_0]
hbm_memory_subsystem::force_kernel_port [get_bd_intf_pins /dpu_2/DPU_AXI_1] 6 1 [get_bd_cells hmss_0]
hbm_memory_subsystem::force_kernel_port [get_bd_intf_pins /dpu_2/DPU_AXI_4] 7 1 [get_bd_cells hmss_0]
hbm_memory_subsystem::force_kernel_port [get_bd_intf_pins /dpu_0/DPU_AXI_W0] 10 1 [get_bd_cells hmss_0]
hbm_memory_subsystem::force_kernel_port [get_bd_intf_pins /dpu_0/DPU_AXI_W1] 11 1 [get_bd_cells hmss_0]
hbm_memory_subsystem::force_kernel_port [get_bd_intf_pins /dpu_1/DPU_AXI_W0] 12 1 [get_bd_cells hmss_0]
hbm_memory_subsystem::force_kernel_port [get_bd_intf_pins /dpu_1/DPU_AXI_W1] 13 1 [get_bd_cells hmss_0]
hbm_memory_subsystem::force_kernel_port [get_bd_intf_pins /dpu_2/DPU_AXI_W0] 14 1 [get_bd_cells hmss_0]
hbm_memory_subsystem::force_kernel_port [get_bd_intf_pins /dpu_2/DPU_AXI_W1] 15 1 [get_bd_cells hmss_0]
hbm_memory_subsystem::force_kernel_port [get_bd_intf_pins /dpu_0/DPU_AXI_I0] 16 1 [get_bd_cells hmss_0]
hbm_memory_subsystem::force_kernel_port [get_bd_intf_pins /dpu_1/DPU_AXI_I0] 17 1 [get_bd_cells hmss_0]
hbm_memory_subsystem::force_kernel_port [get_bd_intf_pins /dpu_2/DPU_AXI_I0] 18 1 [get_bd_cells hmss_0]
hbm_memory_subsystem::force_kernel_port [get_bd_intf_pins /dpu_0/DPU_AXI_1] 19 1 [get_bd_cells hmss_0]
hbm_memory_subsystem::force_kernel_port [get_bd_intf_pins /dpu_0/DPU_AXI_2] 20 1 [get_bd_cells hmss_0]
hbm_memory_subsystem::force_kernel_port [get_bd_intf_pins /dpu_1/DPU_AXI_2] 21 1 [get_bd_cells hmss_0]
hbm_memory_subsystem::force_kernel_port [get_bd_intf_pins /dpu_1/DPU_AXI_3] 22 1 [get_bd_cells hmss_0]
hbm_memory_subsystem::force_kernel_port [get_bd_intf_pins /dpu_2/DPU_AXI_2] 23 1 [get_bd_cells hmss_0]
hbm_memory_subsystem::force_kernel_port [get_bd_intf_pins /dpu_2/DPU_AXI_3] 24 1 [get_bd_cells hmss_0]


set one [get_bd_intf_pins /dpu_1/DPU_AXI_0]
set another [get_bd_intf_pins -of [get_bd_intf_nets -of $one] -filter {MODE==Slave}]
delete_bd_objs [get_bd_intf_nets -of [get_bd_intf_pins /dpu_1/DPU_AXI_0]]
create_bd_cell -type ip -vlnv xilinx.com:ip:axi_register_slice:2.1 axi_register_slice_dpu_1_0
connect_bd_intf_net $one [get_bd_intf_pins axi_register_slice_dpu_1_0/S_AXI]
connect_bd_intf_net [get_bd_intf_pins axi_register_slice_dpu_1_0/M_AXI] $another
connect_bd_net [get_bd_pins /dpu_1/ap_clk] [get_bd_pins axi_register_slice_dpu_1_0/aclk]
connect_bd_net [get_bd_pins axi_register_slice_dpu_1_0/aresetn] [get_bd_pins /dpu_1/ap_rst_n]
set one [get_bd_intf_pins /dpu_1/DPU_AXI_1]
set another [get_bd_intf_pins -of [get_bd_intf_nets -of $one] -filter {MODE==Slave}]
delete_bd_objs [get_bd_intf_nets -of [get_bd_intf_pins /dpu_1/DPU_AXI_1]]
create_bd_cell -type ip -vlnv xilinx.com:ip:axi_register_slice:2.1 axi_register_slice_dpu_1_1
connect_bd_intf_net $one [get_bd_intf_pins axi_register_slice_dpu_1_1/S_AXI]
connect_bd_intf_net [get_bd_intf_pins axi_register_slice_dpu_1_1/M_AXI] $another
connect_bd_net [get_bd_pins /dpu_1/ap_clk] [get_bd_pins axi_register_slice_dpu_1_1/aclk]
connect_bd_net [get_bd_pins axi_register_slice_dpu_1_1/aresetn] [get_bd_pins /dpu_1/ap_rst_n]
set one [get_bd_intf_pins /dpu_1/DPU_AXI_4]
set another [get_bd_intf_pins -of [get_bd_intf_nets -of $one] -filter {MODE==Slave}]
delete_bd_objs [get_bd_intf_nets -of [get_bd_intf_pins /dpu_1/DPU_AXI_4]]
create_bd_cell -type ip -vlnv xilinx.com:ip:axi_register_slice:2.1 axi_register_slice_dpu_1_4
connect_bd_intf_net $one [get_bd_intf_pins axi_register_slice_dpu_1_4/S_AXI]
connect_bd_intf_net [get_bd_intf_pins axi_register_slice_dpu_1_4/M_AXI] $another
connect_bd_net [get_bd_pins /dpu_1/ap_clk] [get_bd_pins axi_register_slice_dpu_1_4/aclk]
connect_bd_net [get_bd_pins axi_register_slice_dpu_1_4/aresetn] [get_bd_pins /dpu_1/ap_rst_n]
set one [get_bd_intf_pins /dpu_2/DPU_AXI_0]
set another [get_bd_intf_pins -of [get_bd_intf_nets -of $one] -filter {MODE==Slave}]
delete_bd_objs [get_bd_intf_nets -of [get_bd_intf_pins /dpu_2/DPU_AXI_0]]
create_bd_cell -type ip -vlnv xilinx.com:ip:axi_register_slice:2.1 axi_register_slice_0
connect_bd_intf_net $one [get_bd_intf_pins axi_register_slice_0/S_AXI]
connect_bd_intf_net [get_bd_intf_pins axi_register_slice_0/M_AXI] $another
connect_bd_net [get_bd_pins /dpu_2/ap_clk] [get_bd_pins axi_register_slice_0/aclk]
connect_bd_net [get_bd_pins axi_register_slice_0/aresetn] [get_bd_pins /dpu_2/ap_rst_n]
set one [get_bd_intf_pins /dpu_2/DPU_AXI_1]
set another [get_bd_intf_pins -of [get_bd_intf_nets -of $one] -filter {MODE==Slave}]
delete_bd_objs [get_bd_intf_nets -of [get_bd_intf_pins /dpu_2/DPU_AXI_1]]
create_bd_cell -type ip -vlnv xilinx.com:ip:axi_register_slice:2.1 axi_register_slice_1
connect_bd_intf_net $one [get_bd_intf_pins axi_register_slice_1/S_AXI]
connect_bd_intf_net [get_bd_intf_pins axi_register_slice_1/M_AXI] $another
connect_bd_net [get_bd_pins /dpu_2/ap_clk] [get_bd_pins axi_register_slice_1/aclk]
connect_bd_net [get_bd_pins axi_register_slice_1/aresetn] [get_bd_pins /dpu_2/ap_rst_n]
set one [get_bd_intf_pins /dpu_2/DPU_AXI_4]
set another [get_bd_intf_pins -of [get_bd_intf_nets -of $one] -filter {MODE==Slave}]
delete_bd_objs [get_bd_intf_nets -of [get_bd_intf_pins /dpu_2/DPU_AXI_4]]
create_bd_cell -type ip -vlnv xilinx.com:ip:axi_register_slice:2.1 axi_register_slice_4
connect_bd_intf_net $one [get_bd_intf_pins axi_register_slice_4/S_AXI]
connect_bd_intf_net [get_bd_intf_pins axi_register_slice_4/M_AXI] $another
connect_bd_net [get_bd_pins /dpu_2/ap_clk] [get_bd_pins axi_register_slice_4/aclk]
connect_bd_net [get_bd_pins axi_register_slice_4/aresetn] [get_bd_pins /dpu_2/ap_rst_n]
set one [get_bd_intf_pins /dpu_1/DPU_AXI_W0]
set another [get_bd_intf_pins -of [get_bd_intf_nets -of $one] -filter {MODE==Slave}]
delete_bd_objs [get_bd_intf_nets -of [get_bd_intf_pins /dpu_1/DPU_AXI_W0]]
create_bd_cell -type ip -vlnv xilinx.com:ip:axi_register_slice:2.1 axi_register_slice_dpu_1_W0
connect_bd_intf_net $one [get_bd_intf_pins axi_register_slice_dpu_1_W0/S_AXI]
connect_bd_intf_net [get_bd_intf_pins axi_register_slice_dpu_1_W0/M_AXI] $another
connect_bd_net [get_bd_pins /dpu_1/ap_clk] [get_bd_pins axi_register_slice_dpu_1_W0/aclk]
connect_bd_net [get_bd_pins axi_register_slice_dpu_1_W0/aresetn] [get_bd_pins /dpu_1/ap_rst_n]
set one [get_bd_intf_pins /dpu_1/DPU_AXI_W1]
set another [get_bd_intf_pins -of [get_bd_intf_nets -of $one] -filter {MODE==Slave}]
delete_bd_objs [get_bd_intf_nets -of [get_bd_intf_pins /dpu_1/DPU_AXI_W1]]
create_bd_cell -type ip -vlnv xilinx.com:ip:axi_register_slice:2.1 axi_register_slice_dpu_1_W1
connect_bd_intf_net $one [get_bd_intf_pins axi_register_slice_dpu_1_W1/S_AXI]
connect_bd_intf_net [get_bd_intf_pins axi_register_slice_dpu_1_W1/M_AXI] $another
connect_bd_net [get_bd_pins /dpu_1/ap_clk] [get_bd_pins axi_register_slice_dpu_1_W1/aclk]
connect_bd_net [get_bd_pins axi_register_slice_dpu_1_W1/aresetn] [get_bd_pins /dpu_1/ap_rst_n]
set one [get_bd_intf_pins /dpu_2/DPU_AXI_W0]
set another [get_bd_intf_pins -of [get_bd_intf_nets -of $one] -filter {MODE==Slave}]
delete_bd_objs [get_bd_intf_nets -of [get_bd_intf_pins /dpu_2/DPU_AXI_W0]]
create_bd_cell -type ip -vlnv xilinx.com:ip:axi_register_slice:2.1 axi_register_slice_W0
connect_bd_intf_net $one [get_bd_intf_pins axi_register_slice_W0/S_AXI]
connect_bd_intf_net [get_bd_intf_pins axi_register_slice_W0/M_AXI] $another
connect_bd_net [get_bd_pins /dpu_2/ap_clk] [get_bd_pins axi_register_slice_W0/aclk]
connect_bd_net [get_bd_pins axi_register_slice_W0/aresetn] [get_bd_pins /dpu_2/ap_rst_n]
set one [get_bd_intf_pins /dpu_2/DPU_AXI_W1]
set another [get_bd_intf_pins -of [get_bd_intf_nets -of $one] -filter {MODE==Slave}]
delete_bd_objs [get_bd_intf_nets -of [get_bd_intf_pins /dpu_2/DPU_AXI_W1]]
create_bd_cell -type ip -vlnv xilinx.com:ip:axi_register_slice:2.1 axi_register_slice_W1
connect_bd_intf_net $one [get_bd_intf_pins axi_register_slice_W1/S_AXI]
connect_bd_intf_net [get_bd_intf_pins axi_register_slice_W1/M_AXI] $another
connect_bd_net [get_bd_pins /dpu_2/ap_clk] [get_bd_pins axi_register_slice_W1/aclk]
connect_bd_net [get_bd_pins axi_register_slice_W1/aresetn] [get_bd_pins /dpu_2/ap_rst_n]
set one [get_bd_intf_pins /dpu_2/DPU_AXI_I0]
set another [get_bd_intf_pins -of [get_bd_intf_nets -of $one] -filter {MODE==Slave}]
delete_bd_objs [get_bd_intf_nets -of [get_bd_intf_pins /dpu_2/DPU_AXI_I0]]
create_bd_cell -type ip -vlnv xilinx.com:ip:axi_register_slice:2.1 axi_register_slice_I0
connect_bd_intf_net $one [get_bd_intf_pins axi_register_slice_I0/S_AXI]
connect_bd_intf_net [get_bd_intf_pins axi_register_slice_I0/M_AXI] $another
connect_bd_net [get_bd_pins /dpu_2/ap_clk] [get_bd_pins axi_register_slice_I0/aclk]
connect_bd_net [get_bd_pins axi_register_slice_I0/aresetn] [get_bd_pins /dpu_2/ap_rst_n]
set one [get_bd_intf_pins /dpu_2/DPU_AXI_2]
set another [get_bd_intf_pins -of [get_bd_intf_nets -of $one] -filter {MODE==Slave}]
delete_bd_objs [get_bd_intf_nets -of [get_bd_intf_pins /dpu_2/DPU_AXI_2]]
create_bd_cell -type ip -vlnv xilinx.com:ip:axi_register_slice:2.1 axi_register_slice_2
connect_bd_intf_net $one [get_bd_intf_pins axi_register_slice_2/S_AXI]
connect_bd_intf_net [get_bd_intf_pins axi_register_slice_2/M_AXI] $another
connect_bd_net [get_bd_pins /dpu_2/ap_clk] [get_bd_pins axi_register_slice_2/aclk]
connect_bd_net [get_bd_pins axi_register_slice_2/aresetn] [get_bd_pins /dpu_2/ap_rst_n]
set one [get_bd_intf_pins /dpu_2/DPU_AXI_3]
set another [get_bd_intf_pins -of [get_bd_intf_nets -of $one] -filter {MODE==Slave}]
delete_bd_objs [get_bd_intf_nets -of [get_bd_intf_pins /dpu_2/DPU_AXI_3]]
create_bd_cell -type ip -vlnv xilinx.com:ip:axi_register_slice:2.1 axi_register_slice_3
connect_bd_intf_net $one [get_bd_intf_pins axi_register_slice_3/S_AXI]
connect_bd_intf_net [get_bd_intf_pins axi_register_slice_3/M_AXI] $another
connect_bd_net [get_bd_pins /dpu_2/ap_clk] [get_bd_pins axi_register_slice_3/aclk]
connect_bd_net [get_bd_pins axi_register_slice_3/aresetn] [get_bd_pins /dpu_2/ap_rst_n]



assign_bd_address [get_bd_addr_segs {dpu_2/s_axi_control/reg0}]

set_param bd.hooks.addr.debug_scoped_use_ms_name true
validate_bd_design -force
