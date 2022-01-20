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
