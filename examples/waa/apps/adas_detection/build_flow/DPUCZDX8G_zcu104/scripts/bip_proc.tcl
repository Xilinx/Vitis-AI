#****************************************************************
# bip_pagespec
#****************************************************************
proc bip_pagespec { spec_name } {
  return [ipgui::get_pagespec -name $spec_name -component [ipx::current_core]]
}

#****************************************************************
# bip_add_files
#****************************************************************
proc bip_add_files { file_group file_paths list_property } {
  set proj_filegroup [ipx::get_file_groups $file_group -of_objects [ipx::current_core]]
  if {$proj_filegroup == {}} {
    set proj_filegroup [ipx::add_file_group -type $file_group "" [ipx::current_core]]
  }
  foreach path $file_paths {
    set f [ipx::add_file $path $proj_filegroup]
    set_property -dict $list_property $f
  }
}

#****************************************************************
# bip_add_bd
#****************************************************************
proc bip_add_bd { path_to_scripts path_to_packaged {path_to_src src} { bd_file_name bd.tcl } } {
  file mkdir $path_to_packaged
  file mkdir $path_to_packaged/$path_to_src
  file copy -force $path_to_scripts/$bd_file_name $path_to_packaged/$path_to_src/.
  bip_add_files xilinx_blockdiagram $path_to_src/$bd_file_name [list type tclSource]
}

#****************************************************************
# bip_get_bus_interfaces
#****************************************************************
proc bip_get_bus_interfaces { bus_name } {
  return [ipx::get_bus_interfaces $bus_name -of_objects [ipx::current_core]]
}

#****************************************************************
# bip_set_bus_interfaces
#****************************************************************
proc bip_set_bus_interfaces {} {
  set_property value {} [ipx::get_bus_parameters ASSOCIATED_BUSIF -of_objects [bip_get_bus_interfaces aclk  ]]
  foreach i [filter [bip_get_bus_interfaces * ] {BUS_TYPE_NAME =~ *axi*}] {
    set i_type    [get_property BUS_TYPE_NAME   $i]
    set i_mode    [get_property INTERFACE_MODE  $i]
    set i_name    [get_property NAME            $i]
    set i_clk     {aclk}
    ipx::associate_bus_interfaces -busif $i_name -clock $i_clk [ipx::current_core]
  }
}

#****************************************************************
# bip_set_bus_enablement_dependency
#****************************************************************
proc bip_set_bus_enablement_dependency {} {
  for {set cnt 0} {$cnt<2} {incr cnt} {
    set act_cnt [expr {($cnt<1)?(0):(2)}]
    set_property enablement_dependency  [join [list {$LOAD_PARALLEL>} $cnt] {}] [bip_get_bus_interfaces M_AXI_HP$act_cnt]
  }
}
