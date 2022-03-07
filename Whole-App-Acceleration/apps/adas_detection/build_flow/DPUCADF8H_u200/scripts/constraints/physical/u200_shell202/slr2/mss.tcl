puts "Start to source [info script]"

if { $VIVADO_VER=="201802" } {
    delete_pblocks [get_pblocks pfm_top_i_dynamic_region_memory_subsystem_inst_pblock_ddr4_mem00]
    set_property CELL_BLOAT_FACTOR medium [get_cells { pfm_top_i/dynamic_region/memory_subsystem/inst/interconnect/interconnect_ddr4_mem00/inst }]
    set_property CELL_BLOAT_FACTOR medium [get_cells { pfm_top_i/dynamic_region/memory_subsystem/inst/memory/ddr4_mem00/inst/u_ddr4_mem_intfc }]
    set_property SOFT_HLUTNM "" [get_cells -hierarchical -filter { PRIMITIVE_TYPE =~ CLB.LUT.* && NAME =~  "*pfm_top_i/dynamic_region*" && SOFT_HLUTNM != "" } ]
}
