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

##################
# Primary Clocks #
##################

# create_clock -period 3.367 -name dp_video_refclk [get_pins */zynq_ultra_ps_e_0/inst/PS8_i/DPVIDEOREFCLK]

################
# Clock Groups #
################

# There is no defined phase relationship, hence they are treated as asynchronous
#set_clock_groups -asynchronous -group [get_clocks -of [get_pins */clk_wiz_1/inst/mmcme3_adv_inst/CLKOUT0]] -group [get_clocks -of [get_pins */clk_wiz_1/inst/mmcme3_adv_inst/CLKOUT1]] -group [get_clocks -of [get_pins */clk_wiz_1/inst/mmcme3_adv_inst/CLKOUT2]] -group [get_clocks -of [get_pins */clk_wiz_1/inst/mmcme3_adv_inst/CLKOUT3]] -group [get_clocks -of [get_pins */clk_wiz_1/inst/mmcme3_adv_inst/CLKOUT4]] -group [get_clocks -of [get_pins */vid_phy_controller_0/inst/gt_usrclk_source_inst/tx_mmcm.txoutclk_mmcm0_i/mmcm_adv_inst/CLKOUT2]] -group dp_video_refclk

# A BUFGMUX selects between DP and HDMI video clock to couple the TPG with the desired display interface
# The two clocks are exclusive since they don't exist at the same time
# set_clock_groups -logically_exclusive -group dp_video_refclk -group [get_clocks -of [get_pins */vid_phy_controller_0/inst/gt_usrclk_source_inst/tx_mmcm.txoutclk_mmcm0_i/mmcm_adv_inst/CLKOUT2]]

###############
# False Paths #
###############

# False path constraint for synchronizer
# set_false_path -to [get_pins -hier *cdc_to*/D]
