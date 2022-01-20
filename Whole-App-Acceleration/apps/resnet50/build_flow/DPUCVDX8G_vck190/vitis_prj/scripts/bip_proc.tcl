# /*                                                                         
# * Copyright 2019 Xilinx Inc.                                               
# *                                                                          
# * Licensed under the Apache License, Version 2.0 (the "License");          
# * you may not use this file except in compliance with the License.         
# * You may obtain a copy of the License at                                  
# *                                                                          
# *    http://www.apache.org/licenses/LICENSE-2.0                            
# *                                                                          
# * Unless required by applicable law or agreed to in writing, software      
# * distributed under the License is distributed on an "AS IS" BASIS,        
# * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. 
# * See the License for the specific language governing permissions and      
# * limitations under the License.                                           
# */  

#****************************************************************
# bip_pagespec
#****************************************************************
proc bip_pagespec { spec_name } {
  return [ipgui::get_pagespec -name $spec_name -component [ipx::current_core]]
}

#****************************************************************
# bip_add_user_param
#****************************************************************
proc bip_add_user_param { param_name } {
  ipx::add_user_parameter $param_name [ipx::current_core]
}

#****************************************************************
# bip_get_user_param
#****************************************************************
proc bip_get_user_param { param_name } {
  ipx::get_user_parameters $param_name -of_objects [ipx::current_core]
}

#****************************************************************
# bip_set_user_parameter
#****************************************************************
proc bip_set_user_parameter {} {
  bip_add_user_param TIMESTAMP_ENA
  set_property value_resolve_type {user}                                                    [bip_get_user_param TIMESTAMP_ENA]
  set_property value_format       {long}                                                    [bip_get_user_param TIMESTAMP_ENA]
  set_property value              {1}                                                       [bip_get_user_param TIMESTAMP_ENA]
}

#****************************************************************
# bip_set_wgtbc
#****************************************************************
proc bip_set_wgtbc {} {
  bip_add_user_param WGTBC_N
  set_property enablement_value   {false}                                                   [bip_get_user_param WGTBC_N]
  set_property value_resolve_type {user}                                                    [bip_get_user_param WGTBC_N]
  set_property value_format       {long}                                                    [bip_get_user_param WGTBC_N]
  set_property value_tcl_expr     {expr ($BATCH_N+$BATCH_SHAREWGT_N-1)/$BATCH_SHAREWGT_N}   [bip_get_user_param WGTBC_N]
  set_property value              {1}                                                       [bip_get_user_param WGTBC_N]
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
  set_property value {} [ipx::get_bus_parameters ASSOCIATED_BUSIF -of_objects [bip_get_bus_interfaces m_axi_aclk  ]]
  foreach i [filter [bip_get_bus_interfaces * ] {BUS_TYPE_NAME =~ *axi*}] {
    set i_type    [get_property BUS_TYPE_NAME   $i]
    set i_mode    [get_property INTERFACE_MODE  $i]
    set i_name    [get_property NAME            $i]
    set i_clk     {m_axi_aclk}
    if { ($i_mode == {slave} ) && ($i_type == {aximm} )  } {
      # s_axi_control
    } else {
      ipx::associate_bus_interfaces -busif $i_name -clock $i_clk [ipx::current_core]
    }
  }
}

#****************************************************************
# bip_set_bus_enablement_dependency
#****************************************************************
proc bip_set_bus_enablement_dependency {} {
  for {set cnt 0} {$cnt< 70} {incr cnt} {
    set_property enablement_dependency  [join [list {$BATCH_N*$ARCH_PP /2*$ARCH_ICP/8>} $cnt] {}] [bip_get_bus_interfaces M[format %02d $cnt]_IFM_AXIS]
  }
  for {set cnt 0} {$cnt<100} {incr cnt} {
    set_property enablement_dependency  [join [list {$WGTBC_N*$ARCH_OCP/8*$ARCH_ICP/8>} $cnt] {}] [bip_get_bus_interfaces M[format %02d $cnt]_WGT_AXIS]
  }
  for {set cnt 100} {$cnt<130} {incr cnt} {
    set_property enablement_dependency  [join [list {$WGTBC_N*$ARCH_OCP/8*$ARCH_ICP/8>} $cnt] {}] [bip_get_bus_interfaces M[format %03d $cnt]_WGT_AXIS]
  }
  for {set cnt 0} {$cnt<100} {incr cnt} {
    set_property enablement_dependency  [join [list {$BATCH_N*$ARCH_PP /2*$ARCH_OCP/8>} $cnt] {}] [bip_get_bus_interfaces S[format %02d $cnt]_OFM_AXIS]
  }
  for {set cnt 100} {$cnt<260} {incr cnt} {
    set_property enablement_dependency  [join [list {$BATCH_N*$ARCH_PP /2*$ARCH_OCP/8>} $cnt] {}] [bip_get_bus_interfaces S[format %03d $cnt]_OFM_AXIS]
  }
  for {set cnt 0} {$cnt<40} {incr cnt} {
    set_property enablement_dependency  [join [list {$BATCH_N*$LOAD_PARALLEL_IMG>} $cnt]      {}] [bip_get_bus_interfaces M[format %02d $cnt]_IMG_AXI ]
  }
  for {set cnt 0} {$cnt<4 } {incr cnt} {
    set_property enablement_dependency  [join [list {         $LOAD_PARALLEL_WGT>} $cnt]      {}] [bip_get_bus_interfaces M[format %02d $cnt]_WGT_AXI ]
  }
  for {set cnt 0} {$cnt<1} {incr cnt} {
    set_property enablement_dependency  [join [list {        $LOAD_PARALLEL_BIAS>} $cnt]      {}] [bip_get_bus_interfaces M[format %02d $cnt]_BIAS_AXI]
  }
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
