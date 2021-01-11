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

}

proc post_config_ip {cellpath otherInfo} {
  set ip [get_bd_cells $cellpath]
#  bip_set_time_stamp $ip

}

proc pre_propagate {cellpath otherInfo} {
  set ip [get_bd_cells $cellpath]

}

proc propagate {cellpath otherInfo} {
  set ip [get_bd_cells $cellpath]
  set_property "CONFIG.M_AXI_FREQ_MHZ"    [expr round(([get_property CONFIG.FREQ_HZ [get_bd_pins "$ip/aclk"]]+0.0)/1000000)]  $ip
  set_property "CONFIG.S_AXI_FREQ_MHZ"    [expr round(([get_property CONFIG.FREQ_HZ [get_bd_pins "$ip/aclk"]]+0.0)/1000000)]  $ip

}

proc post_propagate {cellpath otherInfo} {
  set ip [get_bd_cells $cellpath]
#  bip_set_time_stamp $ip

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

# 

