create_pblock pblock_dynamic_HBM_RS_S
create_pblock pblock_dynamic_HBM_RS_M
resize_pblock [get_pblocks pblock_dynamic_HBM_RS_S] -add {CLOCKREGION_X0Y4:CLOCKREGION_X6Y5}
resize_pblock [get_pblocks pblock_dynamic_HBM_RS_M] -add {CLOCKREGION_X0Y2:CLOCKREGION_X6Y3}
set_property PARENT pblock_dynamic_SLR0 [get_pblocks pblock_dynamic_HBM_RS_M]
set_property PARENT pblock_dynamic_SLR1 [get_pblocks pblock_dynamic_HBM_RS_S]



if {[llength [get_cells "level0_i/ulp/hmss_0/inst/path_3/interconnect*/inst"]] > 0} {
	add_cells_to_pblock [get_pblocks pblock_dynamic_HBM_RS_S] [get_cells "level0_i/ulp/hmss_0/inst/path_3/interconnect*/inst/s00_entry_pipeline"]
	add_cells_to_pblock [get_pblocks pblock_dynamic_HBM_RS_M] [get_cells "level0_i/ulp/hmss_0/inst/path_3/interconnect*/inst/m00_exit_pipeline"]
} else { 
	send_msg_id {HBM_RS_PBLOCK 1-2} INFO "level0_i/ulp/hmss_0/inst/path_3 neither interconnect nor switch found !"
} 
if {[llength [get_cells "level0_i/ulp/hmss_0/inst/path_4/interconnect*/inst"]] > 0} {
	add_cells_to_pblock [get_pblocks pblock_dynamic_HBM_RS_S] [get_cells "level0_i/ulp/hmss_0/inst/path_4/interconnect*/inst/s00_entry_pipeline"]
	add_cells_to_pblock [get_pblocks pblock_dynamic_HBM_RS_M] [get_cells "level0_i/ulp/hmss_0/inst/path_4/interconnect*/inst/m00_exit_pipeline"]
} else { 
	send_msg_id {HBM_RS_PBLOCK 1-2} INFO "level0_i/ulp/hmss_0/inst/path_4 neither interconnect nor switch found !"
} 
if {[llength [get_cells "level0_i/ulp/hmss_0/inst/path_10/interconnect*/inst"]] > 0} {
	add_cells_to_pblock [get_pblocks pblock_dynamic_HBM_RS_S] [get_cells "level0_i/ulp/hmss_0/inst/path_10/interconnect*/inst/s00_entry_pipeline"]
	add_cells_to_pblock [get_pblocks pblock_dynamic_HBM_RS_M] [get_cells "level0_i/ulp/hmss_0/inst/path_10/interconnect*/inst/m00_exit_pipeline"]
} else { 
	send_msg_id {HBM_RS_PBLOCK 1-2} INFO "level0_i/ulp/hmss_0/inst/path_10 neither interconnect nor switch found !"
} 
if {[llength [get_cells "level0_i/ulp/hmss_0/inst/path_11/interconnect*/inst"]] > 0} {
	add_cells_to_pblock [get_pblocks pblock_dynamic_HBM_RS_S] [get_cells "level0_i/ulp/hmss_0/inst/path_11/interconnect*/inst/s00_entry_pipeline"]
	add_cells_to_pblock [get_pblocks pblock_dynamic_HBM_RS_M] [get_cells "level0_i/ulp/hmss_0/inst/path_11/interconnect*/inst/m00_exit_pipeline"]
} else { 
	send_msg_id {HBM_RS_PBLOCK 1-2} INFO "level0_i/ulp/hmss_0/inst/path_11 neither interconnect nor switch found !"
} 
if {[llength [get_cells "level0_i/ulp/hmss_0/inst/path_20/interconnect*/inst"]] > 0} {
	add_cells_to_pblock [get_pblocks pblock_dynamic_HBM_RS_S] [get_cells "level0_i/ulp/hmss_0/inst/path_20/interconnect*/inst/s00_entry_pipeline"]
	add_cells_to_pblock [get_pblocks pblock_dynamic_HBM_RS_M] [get_cells "level0_i/ulp/hmss_0/inst/path_20/interconnect*/inst/m00_exit_pipeline"]
} else { 
	send_msg_id {HBM_RS_PBLOCK 1-2} INFO "level0_i/ulp/hmss_0/inst/path_20 neither interconnect nor switch found !"
} 
if {[llength [get_cells "level0_i/ulp/hmss_0/inst/path_21/interconnect*/inst"]] > 0} {
	add_cells_to_pblock [get_pblocks pblock_dynamic_HBM_RS_S] [get_cells "level0_i/ulp/hmss_0/inst/path_21/interconnect*/inst/s00_entry_pipeline"]
	add_cells_to_pblock [get_pblocks pblock_dynamic_HBM_RS_M] [get_cells "level0_i/ulp/hmss_0/inst/path_21/interconnect*/inst/m00_exit_pipeline"]
} else { 
	send_msg_id {HBM_RS_PBLOCK 1-2} INFO "level0_i/ulp/hmss_0/inst/path_21 neither interconnect nor switch found !"
} 
if {[llength [get_cells "level0_i/ulp/hmss_0/inst/path_26/interconnect*/inst"]] > 0} {
	add_cells_to_pblock [get_pblocks pblock_dynamic_HBM_RS_S] [get_cells "level0_i/ulp/hmss_0/inst/path_26/interconnect*/inst/s00_entry_pipeline"]
	add_cells_to_pblock [get_pblocks pblock_dynamic_HBM_RS_M] [get_cells "level0_i/ulp/hmss_0/inst/path_26/interconnect*/inst/m00_exit_pipeline"]
} else { 
	send_msg_id {HBM_RS_PBLOCK 1-2} INFO "level0_i/ulp/hmss_0/inst/path_26 neither interconnect nor switch found !"
} 
