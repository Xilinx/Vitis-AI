startgroup
create_bd_cell -type ip -vlnv xilinx.com:ip:axi_interconnect:2.1 axi_interconnect_0
replace_bd_cell -preserve_configuration -preserve_name [get_bd_cells /interconnect_axilite_user_slr0] [get_bd_cells /axi_interconnect_0]
delete_bd_objs [get_bd_cells interconnect_axilite_user_slr0_old1]
endgroup
