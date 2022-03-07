################
# Clock Groups #
################
#
create_clock -period 3.367 -name hdmi_tx_clk [get_ports TX_REFCLK_P_IN]
create_clock -period 10.000 -name audio_ref_clk [get_ports CLK_IN_AUDIO_clk_p]

set_false_path -from [get_cells -hier -filter name=~*HDMI_ACR_CTRL_AXI_INST/rEnab_ACR_reg] -to [get_cells -hier -filter {name=~*aud_enab_acr_sync_reg[0]}]
set_false_path -from [get_cells -hier -filter name=~*HDMI_ACR_CTRL_AXI_INST/rACR_Sel_reg] -to [get_cells -hier -filter {name=~*aud_acr_sel_sync_reg[0]}]
set_false_path -from [get_cells -hier -filter name=~*HDMI_ACR_CTRL_AXI_INST/rTMDSClkRatio_reg] -to [get_cells -hier -filter {name=~*aud_tmdsclkratio_sync_reg[0]}]
set_false_path -from [get_cells -hier -filter name=~*PULSE_CLKCROSS_INST/rIn_Toggle_reg] -to [get_cells -hier -filter {name=~*PULSE_CLKCROSS_INST/rOut_Sync_reg[0]}]
set_false_path -from [get_cells -hier -filter name=~*HDMI_ACR_CTRL_AXI_INST/rAud_Reset_reg] -to [get_cells -hier -filter {name=~*aud_rst_chain_reg[*]}]
set_false_path -from [get_cells -hier -filter {name=~*NVAL_CLKCROSS_INST/rIn_Data_reg[*]}] -to [get_cells -hier -filter {name=~*NVAL_CLKCROSS_INST/rOut_Data_reg[*]}]
set_false_path -from [get_cells -hier -filter name=~*NVAL_CLKCROSS_INST/rIn_DValid_reg] -to [get_cells -hier -filter {name=~*NVAL_CLKCROSS_INST/rOut_DValid_Sync_reg[0]}]
set_false_path -from [get_cells -hier -filter name=~*NVAL_CLKCROSS_INST/rOut_ACK_reg] -to [get_cells -hier -filter {name=~*NVAL_CLKCROSS_INST/rIn_ACK_Sync_reg[0]}]
set_false_path -from [get_cells -hier -filter {name=~*CTS_CLKCROSS_ACLK_INST/rIn_Data_reg[*]}] -to [get_cells -hier -filter {name=~*CTS_CLKCROSS_ACLK_INST/rOut_Data_reg[*]}]
set_false_path -from [get_cells -hier -filter name=~*CTS_CLKCROSS_ACLK_INST/rIn_DValid_reg] -to [get_cells -hier -filter {name=~*CTS_CLKCROSS_ACLK_INST/rOut_DValid_Sync_reg[0]}]
set_false_path -from [get_cells -hier -filter name=~*CTS_CLKCROSS_ACLK_INST/rOut_ACK_reg] -to [get_cells -hier -filter {name=~*CTS_CLKCROSS_ACLK_INST/rIn_ACK_Sync_reg[0]}]
set_false_path -from [get_cells -hier -filter {name=~*CTS_CLKCROSS_AUD_INST/rIn_Data_reg[*]}] -to [get_cells -hier -filter {name=~*CTS_CLKCROSS_AUD_INST/rOut_Data_reg[*]}]
set_false_path -from [get_cells -hier -filter name=~*CTS_CLKCROSS_AUD_INST/rIn_DValid_reg] -to [get_cells -hier -filter {name=~*CTS_CLKCROSS_AUD_INST/rOut_DValid_Sync_reg[0]}]
set_false_path -from [get_cells -hier -filter name=~*CTS_CLKCROSS_AUD_INST/rOut_ACK_reg] -to [get_cells -hier -filter {name=~*CTS_CLKCROSS_AUD_INST/rIn_ACK_Sync_reg[0]}]

