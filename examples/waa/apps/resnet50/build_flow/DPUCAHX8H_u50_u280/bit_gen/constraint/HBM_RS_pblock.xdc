create_pblock pblock_dynamic_HBM_RS_S
create_pblock pblock_dynamic_HBM_RS_M
resize_pblock [get_pblocks pblock_dynamic_HBM_RS_S] -add {CLOCKREGION_X0Y4:CLOCKREGION_X6Y5}
resize_pblock [get_pblocks pblock_dynamic_HBM_RS_M] -add {CLOCKREGION_X0Y2:CLOCKREGION_X6Y3}
set_property PARENT pblock_dynamic_SLR0 [get_pblocks pblock_dynamic_HBM_RS_M]
set_property PARENT pblock_dynamic_SLR1 [get_pblocks pblock_dynamic_HBM_RS_S]



if {[llength [get_cells "pfm_top_i/dynamic_region/hmss_0/inst/path_5/interconnect*/inst"]] > 0} {
	add_cells_to_pblock [get_pblocks pblock_dynamic_HBM_RS_S] [get_cells "pfm_top_i/dynamic_region/hmss_0/inst/path_5/interconnect*/inst/s00_entry_pipeline"]
	add_cells_to_pblock [get_pblocks pblock_dynamic_HBM_RS_M] [get_cells "pfm_top_i/dynamic_region/hmss_0/inst/path_5/interconnect*/inst/m00_exit_pipeline"]
} else { 
	send_msg_id {HBM_RS_PBLOCK 1-2} INFO "pfm_top_i/dynamic_region/hmss_0/inst/path_5 neither interconnect nor switch found !"
} 
if {[llength [get_cells "pfm_top_i/dynamic_region/hmss_0/inst/path_6/interconnect*/inst"]] > 0} {
	add_cells_to_pblock [get_pblocks pblock_dynamic_HBM_RS_S] [get_cells "pfm_top_i/dynamic_region/hmss_0/inst/path_6/interconnect*/inst/s00_entry_pipeline"]
	add_cells_to_pblock [get_pblocks pblock_dynamic_HBM_RS_M] [get_cells "pfm_top_i/dynamic_region/hmss_0/inst/path_6/interconnect*/inst/m00_exit_pipeline"]
} else { 
	send_msg_id {HBM_RS_PBLOCK 1-2} INFO "pfm_top_i/dynamic_region/hmss_0/inst/path_6 neither interconnect nor switch found !"
} 
if {[llength [get_cells "pfm_top_i/dynamic_region/hmss_0/inst/path_7/interconnect*/inst"]] > 0} {
	add_cells_to_pblock [get_pblocks pblock_dynamic_HBM_RS_S] [get_cells "pfm_top_i/dynamic_region/hmss_0/inst/path_7/interconnect*/inst/s00_entry_pipeline"]
	add_cells_to_pblock [get_pblocks pblock_dynamic_HBM_RS_M] [get_cells "pfm_top_i/dynamic_region/hmss_0/inst/path_7/interconnect*/inst/m00_exit_pipeline"]
} else { 
	send_msg_id {HBM_RS_PBLOCK 1-2} INFO "pfm_top_i/dynamic_region/hmss_0/inst/path_7 neither interconnect nor switch found !"
} 
if {[llength [get_cells "pfm_top_i/dynamic_region/hmss_0/inst/path_14/interconnect*/inst"]] > 0} {
	add_cells_to_pblock [get_pblocks pblock_dynamic_HBM_RS_S] [get_cells "pfm_top_i/dynamic_region/hmss_0/inst/path_14/interconnect*/inst/s00_entry_pipeline"]
	add_cells_to_pblock [get_pblocks pblock_dynamic_HBM_RS_M] [get_cells "pfm_top_i/dynamic_region/hmss_0/inst/path_14/interconnect*/inst/m00_exit_pipeline"]
} else { 
	send_msg_id {HBM_RS_PBLOCK 1-2} INFO "pfm_top_i/dynamic_region/hmss_0/inst/path_14 neither interconnect nor switch found !"
} 
if {[llength [get_cells "pfm_top_i/dynamic_region/hmss_0/inst/path_15/interconnect*/inst"]] > 0} {
	add_cells_to_pblock [get_pblocks pblock_dynamic_HBM_RS_S] [get_cells "pfm_top_i/dynamic_region/hmss_0/inst/path_15/interconnect*/inst/s00_entry_pipeline"]
	add_cells_to_pblock [get_pblocks pblock_dynamic_HBM_RS_M] [get_cells "pfm_top_i/dynamic_region/hmss_0/inst/path_15/interconnect*/inst/m00_exit_pipeline"]
} else { 
	send_msg_id {HBM_RS_PBLOCK 1-2} INFO "pfm_top_i/dynamic_region/hmss_0/inst/path_15 neither interconnect nor switch found !"
} 
if {[llength [get_cells "pfm_top_i/dynamic_region/hmss_0/inst/path_18/interconnect*/inst"]] > 0} {
	add_cells_to_pblock [get_pblocks pblock_dynamic_HBM_RS_S] [get_cells "pfm_top_i/dynamic_region/hmss_0/inst/path_18/interconnect*/inst/s00_entry_pipeline"]
	add_cells_to_pblock [get_pblocks pblock_dynamic_HBM_RS_M] [get_cells "pfm_top_i/dynamic_region/hmss_0/inst/path_18/interconnect*/inst/m00_exit_pipeline"]
} else { 
	send_msg_id {HBM_RS_PBLOCK 1-2} INFO "pfm_top_i/dynamic_region/hmss_0/inst/path_18 neither interconnect nor switch found !"
} 
if {[llength [get_cells "pfm_top_i/dynamic_region/hmss_0/inst/path_23/interconnect*/inst"]] > 0} {
	add_cells_to_pblock [get_pblocks pblock_dynamic_HBM_RS_S] [get_cells "pfm_top_i/dynamic_region/hmss_0/inst/path_23/interconnect*/inst/s00_entry_pipeline"]
	add_cells_to_pblock [get_pblocks pblock_dynamic_HBM_RS_M] [get_cells "pfm_top_i/dynamic_region/hmss_0/inst/path_23/interconnect*/inst/m00_exit_pipeline"]
} else { 
	send_msg_id {HBM_RS_PBLOCK 1-2} INFO "pfm_top_i/dynamic_region/hmss_0/inst/path_23 neither interconnect nor switch found !"
} 
if {[llength [get_cells "pfm_top_i/dynamic_region/hmss_0/inst/path_24/interconnect*/inst"]] > 0} {
	add_cells_to_pblock [get_pblocks pblock_dynamic_HBM_RS_S] [get_cells "pfm_top_i/dynamic_region/hmss_0/inst/path_24/interconnect*/inst/s00_entry_pipeline"]
	add_cells_to_pblock [get_pblocks pblock_dynamic_HBM_RS_M] [get_cells "pfm_top_i/dynamic_region/hmss_0/inst/path_24/interconnect*/inst/m00_exit_pipeline"]
} else { 
	send_msg_id {HBM_RS_PBLOCK 1-2} INFO "pfm_top_i/dynamic_region/hmss_0/inst/path_24 neither interconnect nor switch found !"
} 
