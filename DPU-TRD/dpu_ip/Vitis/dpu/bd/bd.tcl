# (c) Copyright 2018-2019 Xilinx, Inc. All rights reserved.
#
# This file contains confidential and proprietary information
# of Xilinx, Inc. and is protected under U.S. and
# international copyright and other intellectual property
# laws.
#
# DISCLAIMER
# This disclaimer is not a license and does not grant any
# rights to the materials distributed herewith. Except as
# otherwise provided in a valid license issued to you by
# Xilinx, and to the maximum extent permitted by applicable
# law: (1) THESE MATERIALS ARE MADE AVAILABLE "AS IS" AND
# WITH ALL FAULTS, AND XILINX HEREBY DISCLAIMS ALL WARRANTIES
# AND CONDITIONS, EXPRESS, IMPLIED, OR STATUTORY, INCLUDING
# BUT NOT LIMITED TO WARRANTIES OF MERCHANTABILITY, NON-
# INFRINGEMENT, OR FITNESS FOR ANY PARTICULAR PURPOSE; and
# (2) Xilinx shall not be liable (whether in contract or tort,
# including negligence, or under any other theory of
# liability) for any loss or damage of any kind or nature
# related to, arising under or in connection with these
# materials, including for any direct, or any indirect,
# special, incidental, or consequential loss or damage
# (including loss of data, profits, goodwill, or any type of
# loss or damage suffered as a result of any action brought
# by a third party) even if such damage or loss was
# reasonably foreseeable or Xilinx had been advised of the
# possibility of the same.
#
# CRITICAL APPLICATIONS
# Xilinx products are not designed or intended to be fail-
# safe, or for use in any application requiring fail-safe
# performance, such as life-support or safety devices or
# systems, Class III medical devices, nuclear facilities,
# applications related to the deployment of airbags, or any
# other applications that could lead to death, personal
# injury, or severe property or environmental damage
# (individually and collectively, "Critical
# Applications"). Customer assumes the sole risk and
# liability of any use of Xilinx products in Critical
# Applications, subject only to applicable laws and
# regulations governing limitations on product liability.
#
# THIS COPYRIGHT NOTICE AND DISCLAIMER MUST BE RETAINED AS
# PART OF THIS FILE AT ALL TIMES.
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

# 41d051e7f991c9dacbe8d368b85f57380b272bd2d7dc26e6c472a2d06e70908e

