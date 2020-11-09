#
# Copyright 2019 Xilinx Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
proc init {cellpath otherInfo} {
  set ip [get_bd_cells $cellpath]

  set family [string tolower [get_property FAMILY [get_property PART [current_project]]]]
  if {$family == "zynq"} {
    set_property "CONFIG.AXI_PROTOCOL"            {0}           $ip
    set_property "CONFIG.ARCH_HP_BW"              {2}           $ip
    set_property "CONFIG.VER_CHIP_PART"           {1}           $ip
    set_property "CONFIG.DSP48_VER"               {DSP48E1}     $ip
  } else {
    set_property "CONFIG.AXI_PROTOCOL"            {1}           $ip
    set_property "CONFIG.ARCH_HP_BW"              {3}           $ip
    set_property "CONFIG.VER_CHIP_PART"           {3}           $ip
    set_property "CONFIG.DSP48_VER"               {DSP48E2}     $ip
    set_property "CONFIG.CONV_DSP_CASC_MAX"       {4}           $ip
  }
  set_property "CONFIG.ARCH"                      {4096}        $ip
  set_property "CONFIG.DWCV_ENA"                  {1}           $ip

}

proc post_config_ip {cellpath otherInfo} {
  set ip [get_bd_cells $cellpath]
  if { [get_property CONFIG.S_AXI_CLK_INDEPENDENT $ip] == {1} } {
    bip_set_clk_busif "$ip/m_axi_dpu_aclk" {S_AXI} {rmv}
  } else {
    bip_set_clk_busif "$ip/m_axi_dpu_aclk" {S_AXI} {add}
  }
  bip_set_time_stamp $ip

}

proc pre_propagate {cellpath otherInfo} {
  set ip [get_bd_cells $cellpath]
}

proc propagate {cellpath otherInfo} {
  set ip [get_bd_cells $cellpath]

  set_property "CONFIG.M_AXI_FREQ_MHZ"    [expr round(([get_property CONFIG.FREQ_HZ [get_bd_pins "$ip/m_axi_dpu_aclk"]]+0.0)/1000000)]  $ip
  if { [get_property CONFIG.S_AXI_CLK_INDEPENDENT $ip] == {1} } {
    set_property "CONFIG.S_AXI_FREQ_MHZ"  [expr round(([get_property CONFIG.FREQ_HZ [get_bd_pins "$ip/s_axi_aclk"]]+0.0)/1000000)]  $ip
  } else {
    set_property "CONFIG.S_AXI_FREQ_MHZ"  [expr round(([get_property CONFIG.FREQ_HZ [get_bd_pins "$ip/m_axi_dpu_aclk"]]+0.0)/1000000)]  $ip
  }

}

proc post_propagate {cellpath otherInfo} {
  set ip [get_bd_cells $cellpath]

  bip_set_base_addr $ip
  bip_set_time_stamp $ip

}

proc bip_set_clk_busif { pin_clk busif_name {mode add} } {
  set busifs      [get_property CONFIG.ASSOCIATED_BUSIF [get_bd_pins $pin_clk]]
  set busif_subs  $busif_name
  if { [set has_busif [string match *$busif_subs* $busifs]] == {1} } {
    set busif_subs $busif_name:
    if { [string match *$busif_subs* $busifs] != {1} } {
      set busif_subs :$busif_name
      if { [string match *$busif_subs* $busifs] != {1} } {
        set busif_subs $busif_name
      }
    }
  }
  switch $mode {
    add     { if { $has_busif == {0} } { set busifs $busifs:$busif_name                         } }
    rmv     { if { $has_busif == {1} } { set busifs [string map [list $busif_subs {}] $busifs]  } }
    default {}
  }
  set_property CONFIG.ASSOCIATED_BUSIF $busifs [get_bd_pins $pin_clk]
}

proc bip_set_time_stamp {ip} {
  set ena [get_property "CONFIG.TIMESTAMP_ENA" $ip]
  if { $ena > {0} } {
    set_property "CONFIG.TIME_YEAR"     [expr [scan [clock format [clock seconds] -format %Y] %d] -2000 ] $ip
    set_property "CONFIG.TIME_MONTH"    [expr [scan [clock format [clock seconds] -format %m] %d]       ] $ip
    set_property "CONFIG.TIME_DAY"      [expr [scan [clock format [clock seconds] -format %e] %d]       ] $ip
    set_property "CONFIG.TIME_HOUR"     [expr [scan [clock format [clock seconds] -format %k] %d]       ] $ip
    set_property "CONFIG.TIME_QUARTER"  [expr [scan [clock format [clock seconds] -format %M] %d] /15   ] $ip
  }
}

proc bip_set_base_addr { ip } {
  set seg_slv     [get_bd_addr_segs -addressable -of_objects [get_bd_intf_pins $ip/S_AXI]]
  set seg_as      [get_bd_addr_segs -of_objects $seg_slv]
  set seg_offset  [format "0x%x" [expr [lindex [get_property offset  $seg_as] 0] % (2**32)]]
  set seg_range   [format "0x%x" [expr [lindex [get_property range   $seg_as] 0] % (2**32)]]

  set_property    "CONFIG.S_AXI_SLAVES_BASE_ADDR" $seg_offset $ip
  if { [expr $seg_range] != [expr int(pow(2,24))] } {
    ::bd::send_msg -of $ip -type error -msg_id 1 -text " Invalid range ($seg_range) dectected in address segment $seg_as. It should be 16M. Address segment will not be imported."
  }

}

# 
