# Definitional proc to organize widgets for parameters.
proc init_gui { IPINST } {
  ipgui::add_param $IPINST -name "Component_Name"
  #Adding Page
  set Page_0 [ipgui::add_page $IPINST -name "Page 0" -display_name {Configuration}]
  set_property tooltip {Configuration} ${Page_0}
  ipgui::add_param $IPINST -name "CLK_GATE_EN" -parent ${Page_0}
  set DPU_NUM [ipgui::add_param $IPINST -name "DPU_NUM" -parent ${Page_0} -widget comboBox]
  set_property tooltip {Batch numbers in each core} ${DPU_NUM}
  ipgui::add_param $IPINST -name "ENABLE_AVG_MODE" -parent ${Page_0}
  ipgui::add_param $IPINST -name "ENABLE_CH_OFFSET" -parent ${Page_0}
  ipgui::add_param $IPINST -name "FM_BKG_MERGE" -parent ${Page_0}
  ipgui::add_param $IPINST -name "FM_BKG_MODE" -parent ${Page_0}
  ipgui::add_param $IPINST -name "FM_BKG_SIZE" -parent ${Page_0}
  ipgui::add_param $IPINST -name "FM_LOAD_PER_CORE" -parent ${Page_0}
  ipgui::add_param $IPINST -name "FREQ" -parent ${Page_0} -widget comboBox
  ipgui::add_param $IPINST -name "MISC_PP_N" -parent ${Page_0}
  set NL_LEAKYRELU [ipgui::add_param $IPINST -name "NL_LEAKYRELU" -parent ${Page_0} -widget comboBox]
  set_property tooltip {Leaky-Relu is only used in special Neural Networks, such as Yolo} ${NL_LEAKYRELU}
  ipgui::add_param $IPINST -name "SAVE_PER_ENGINE" -parent ${Page_0}
  ipgui::add_param $IPINST -name "URAM_AS_FM_BUFFER" -parent ${Page_0}


}

proc update_PARAM_VALUE.AXI_SHARE { PARAM_VALUE.AXI_SHARE } {
	# Procedure called to update AXI_SHARE when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.AXI_SHARE { PARAM_VALUE.AXI_SHARE } {
	# Procedure called to validate AXI_SHARE
	return true
}

proc update_PARAM_VALUE.CLK_GATE_EN { PARAM_VALUE.CLK_GATE_EN } {
	# Procedure called to update CLK_GATE_EN when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.CLK_GATE_EN { PARAM_VALUE.CLK_GATE_EN } {
	# Procedure called to validate CLK_GATE_EN
	return true
}

proc update_PARAM_VALUE.CONV_DSP_NUM { PARAM_VALUE.CONV_DSP_NUM } {
	# Procedure called to update CONV_DSP_NUM when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.CONV_DSP_NUM { PARAM_VALUE.CONV_DSP_NUM } {
	# Procedure called to validate CONV_DSP_NUM
	return true
}

proc update_PARAM_VALUE.CONV_WFIFO_TYPE { PARAM_VALUE.CONV_WFIFO_TYPE } {
	# Procedure called to update CONV_WFIFO_TYPE when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.CONV_WFIFO_TYPE { PARAM_VALUE.CONV_WFIFO_TYPE } {
	# Procedure called to validate CONV_WFIFO_TYPE
	return true
}

proc update_PARAM_VALUE.CP { PARAM_VALUE.CP } {
	# Procedure called to update CP when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.CP { PARAM_VALUE.CP } {
	# Procedure called to validate CP
	return true
}

proc update_PARAM_VALUE.C_ADDR_W { PARAM_VALUE.C_ADDR_W } {
	# Procedure called to update C_ADDR_W when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.C_ADDR_W { PARAM_VALUE.C_ADDR_W } {
	# Procedure called to validate C_ADDR_W
	return true
}

proc update_PARAM_VALUE.C_DATA_W { PARAM_VALUE.C_DATA_W } {
	# Procedure called to update C_DATA_W when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.C_DATA_W { PARAM_VALUE.C_DATA_W } {
	# Procedure called to validate C_DATA_W
	return true
}

proc update_PARAM_VALUE.DPU_NUM { PARAM_VALUE.DPU_NUM } {
	# Procedure called to update DPU_NUM when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.DPU_NUM { PARAM_VALUE.DPU_NUM } {
	# Procedure called to validate DPU_NUM
	return true
}

proc update_PARAM_VALUE.DPU_WORK_CNT { PARAM_VALUE.DPU_WORK_CNT } {
	# Procedure called to update DPU_WORK_CNT when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.DPU_WORK_CNT { PARAM_VALUE.DPU_WORK_CNT } {
	# Procedure called to validate DPU_WORK_CNT
	return true
}

proc update_PARAM_VALUE.DPU_WORK_CNT_EN { PARAM_VALUE.DPU_WORK_CNT_EN } {
	# Procedure called to update DPU_WORK_CNT_EN when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.DPU_WORK_CNT_EN { PARAM_VALUE.DPU_WORK_CNT_EN } {
	# Procedure called to validate DPU_WORK_CNT_EN
	return true
}

proc update_PARAM_VALUE.DWCONV_EN { PARAM_VALUE.DWCONV_EN } {
	# Procedure called to update DWCONV_EN when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.DWCONV_EN { PARAM_VALUE.DWCONV_EN } {
	# Procedure called to validate DWCONV_EN
	return true
}

proc update_PARAM_VALUE.ENABLE_AVG_MODE { PARAM_VALUE.ENABLE_AVG_MODE } {
	# Procedure called to update ENABLE_AVG_MODE when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.ENABLE_AVG_MODE { PARAM_VALUE.ENABLE_AVG_MODE } {
	# Procedure called to validate ENABLE_AVG_MODE
	return true
}

proc update_PARAM_VALUE.ENABLE_CH_OFFSET { PARAM_VALUE.ENABLE_CH_OFFSET } {
	# Procedure called to update ENABLE_CH_OFFSET when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.ENABLE_CH_OFFSET { PARAM_VALUE.ENABLE_CH_OFFSET } {
	# Procedure called to validate ENABLE_CH_OFFSET
	return true
}

proc update_PARAM_VALUE.FM_BKG_MERGE { PARAM_VALUE.FM_BKG_MERGE } {
	# Procedure called to update FM_BKG_MERGE when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.FM_BKG_MERGE { PARAM_VALUE.FM_BKG_MERGE } {
	# Procedure called to validate FM_BKG_MERGE
	return true
}

proc update_PARAM_VALUE.FM_BKG_MODE { PARAM_VALUE.FM_BKG_MODE } {
	# Procedure called to update FM_BKG_MODE when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.FM_BKG_MODE { PARAM_VALUE.FM_BKG_MODE } {
	# Procedure called to validate FM_BKG_MODE
	return true
}

proc update_PARAM_VALUE.FM_BKG_SIZE { PARAM_VALUE.FM_BKG_SIZE } {
	# Procedure called to update FM_BKG_SIZE when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.FM_BKG_SIZE { PARAM_VALUE.FM_BKG_SIZE } {
	# Procedure called to validate FM_BKG_SIZE
	return true
}

proc update_PARAM_VALUE.FM_LOAD_PER_CORE { PARAM_VALUE.FM_LOAD_PER_CORE } {
	# Procedure called to update FM_LOAD_PER_CORE when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.FM_LOAD_PER_CORE { PARAM_VALUE.FM_LOAD_PER_CORE } {
	# Procedure called to validate FM_LOAD_PER_CORE
	return true
}

proc update_PARAM_VALUE.FREQ { PARAM_VALUE.FREQ } {
	# Procedure called to update FREQ when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.FREQ { PARAM_VALUE.FREQ } {
	# Procedure called to validate FREQ
	return true
}

proc update_PARAM_VALUE.GP_DATA_BW { PARAM_VALUE.GP_DATA_BW } {
	# Procedure called to update GP_DATA_BW when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.GP_DATA_BW { PARAM_VALUE.GP_DATA_BW } {
	# Procedure called to validate GP_DATA_BW
	return true
}

proc update_PARAM_VALUE.HP_DATA_BW { PARAM_VALUE.HP_DATA_BW } {
	# Procedure called to update HP_DATA_BW when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.HP_DATA_BW { PARAM_VALUE.HP_DATA_BW } {
	# Procedure called to validate HP_DATA_BW
	return true
}

proc update_PARAM_VALUE.IBGRP_N { PARAM_VALUE.IBGRP_N } {
	# Procedure called to update IBGRP_N when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.IBGRP_N { PARAM_VALUE.IBGRP_N } {
	# Procedure called to validate IBGRP_N
	return true
}

proc update_PARAM_VALUE.MISC_PP_N { PARAM_VALUE.MISC_PP_N } {
	# Procedure called to update MISC_PP_N when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.MISC_PP_N { PARAM_VALUE.MISC_PP_N } {
	# Procedure called to validate MISC_PP_N
	return true
}

proc update_PARAM_VALUE.MISC_WFIFO_TYPE { PARAM_VALUE.MISC_WFIFO_TYPE } {
	# Procedure called to update MISC_WFIFO_TYPE when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.MISC_WFIFO_TYPE { PARAM_VALUE.MISC_WFIFO_TYPE } {
	# Procedure called to validate MISC_WFIFO_TYPE
	return true
}

proc update_PARAM_VALUE.NL_LEAKYRELU { PARAM_VALUE.NL_LEAKYRELU } {
	# Procedure called to update NL_LEAKYRELU when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.NL_LEAKYRELU { PARAM_VALUE.NL_LEAKYRELU } {
	# Procedure called to validate NL_LEAKYRELU
	return true
}

proc update_PARAM_VALUE.NL_RATIO { PARAM_VALUE.NL_RATIO } {
	# Procedure called to update NL_RATIO when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.NL_RATIO { PARAM_VALUE.NL_RATIO } {
	# Procedure called to validate NL_RATIO
	return true
}

proc update_PARAM_VALUE.PP { PARAM_VALUE.PP } {
	# Procedure called to update PP when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.PP { PARAM_VALUE.PP } {
	# Procedure called to validate PP
	return true
}

proc update_PARAM_VALUE.SAVE_PER_ENGINE { PARAM_VALUE.SAVE_PER_ENGINE } {
	# Procedure called to update SAVE_PER_ENGINE when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.SAVE_PER_ENGINE { PARAM_VALUE.SAVE_PER_ENGINE } {
	# Procedure called to validate SAVE_PER_ENGINE
	return true
}

proc update_PARAM_VALUE.SEVN_DSP_CTL_0 { PARAM_VALUE.SEVN_DSP_CTL_0 } {
	# Procedure called to update SEVN_DSP_CTL_0 when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.SEVN_DSP_CTL_0 { PARAM_VALUE.SEVN_DSP_CTL_0 } {
	# Procedure called to validate SEVN_DSP_CTL_0
	return true
}

proc update_PARAM_VALUE.SEVN_DSP_CTL_1 { PARAM_VALUE.SEVN_DSP_CTL_1 } {
	# Procedure called to update SEVN_DSP_CTL_1 when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.SEVN_DSP_CTL_1 { PARAM_VALUE.SEVN_DSP_CTL_1 } {
	# Procedure called to validate SEVN_DSP_CTL_1
	return true
}

proc update_PARAM_VALUE.SEVN_DSP_CTL_2 { PARAM_VALUE.SEVN_DSP_CTL_2 } {
	# Procedure called to update SEVN_DSP_CTL_2 when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.SEVN_DSP_CTL_2 { PARAM_VALUE.SEVN_DSP_CTL_2 } {
	# Procedure called to validate SEVN_DSP_CTL_2
	return true
}

proc update_PARAM_VALUE.SEVN_DSP_CTL_3 { PARAM_VALUE.SEVN_DSP_CTL_3 } {
	# Procedure called to update SEVN_DSP_CTL_3 when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.SEVN_DSP_CTL_3 { PARAM_VALUE.SEVN_DSP_CTL_3 } {
	# Procedure called to validate SEVN_DSP_CTL_3
	return true
}

proc update_PARAM_VALUE.SEVN_DSP_CTL_4 { PARAM_VALUE.SEVN_DSP_CTL_4 } {
	# Procedure called to update SEVN_DSP_CTL_4 when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.SEVN_DSP_CTL_4 { PARAM_VALUE.SEVN_DSP_CTL_4 } {
	# Procedure called to validate SEVN_DSP_CTL_4
	return true
}

proc update_PARAM_VALUE.SEVN_DSP_CTL_5 { PARAM_VALUE.SEVN_DSP_CTL_5 } {
	# Procedure called to update SEVN_DSP_CTL_5 when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.SEVN_DSP_CTL_5 { PARAM_VALUE.SEVN_DSP_CTL_5 } {
	# Procedure called to validate SEVN_DSP_CTL_5
	return true
}

proc update_PARAM_VALUE.URAM_AS_FM_BUFFER { PARAM_VALUE.URAM_AS_FM_BUFFER } {
	# Procedure called to update URAM_AS_FM_BUFFER when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.URAM_AS_FM_BUFFER { PARAM_VALUE.URAM_AS_FM_BUFFER } {
	# Procedure called to validate URAM_AS_FM_BUFFER
	return true
}


proc update_MODELPARAM_VALUE.URAM_AS_FM_BUFFER { MODELPARAM_VALUE.URAM_AS_FM_BUFFER PARAM_VALUE.URAM_AS_FM_BUFFER } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.URAM_AS_FM_BUFFER}] ${MODELPARAM_VALUE.URAM_AS_FM_BUFFER}
}

proc update_MODELPARAM_VALUE.FM_BKG_MERGE { MODELPARAM_VALUE.FM_BKG_MERGE PARAM_VALUE.FM_BKG_MERGE } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.FM_BKG_MERGE}] ${MODELPARAM_VALUE.FM_BKG_MERGE}
}

proc update_MODELPARAM_VALUE.FM_BKG_SIZE { MODELPARAM_VALUE.FM_BKG_SIZE PARAM_VALUE.FM_BKG_SIZE } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.FM_BKG_SIZE}] ${MODELPARAM_VALUE.FM_BKG_SIZE}
}

proc update_MODELPARAM_VALUE.FM_BKG_MODE { MODELPARAM_VALUE.FM_BKG_MODE PARAM_VALUE.FM_BKG_MODE } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.FM_BKG_MODE}] ${MODELPARAM_VALUE.FM_BKG_MODE}
}

proc update_MODELPARAM_VALUE.FM_LOAD_PER_CORE { MODELPARAM_VALUE.FM_LOAD_PER_CORE PARAM_VALUE.FM_LOAD_PER_CORE } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.FM_LOAD_PER_CORE}] ${MODELPARAM_VALUE.FM_LOAD_PER_CORE}
}

proc update_MODELPARAM_VALUE.SAVE_PER_ENGINE { MODELPARAM_VALUE.SAVE_PER_ENGINE PARAM_VALUE.SAVE_PER_ENGINE } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.SAVE_PER_ENGINE}] ${MODELPARAM_VALUE.SAVE_PER_ENGINE}
}

proc update_MODELPARAM_VALUE.CLK_GATE_EN { MODELPARAM_VALUE.CLK_GATE_EN PARAM_VALUE.CLK_GATE_EN } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.CLK_GATE_EN}] ${MODELPARAM_VALUE.CLK_GATE_EN}
}

proc update_MODELPARAM_VALUE.DPU_NUM { MODELPARAM_VALUE.DPU_NUM PARAM_VALUE.DPU_NUM } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.DPU_NUM}] ${MODELPARAM_VALUE.DPU_NUM}
}

proc update_MODELPARAM_VALUE.AXI_SHARE { MODELPARAM_VALUE.AXI_SHARE PARAM_VALUE.AXI_SHARE } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.AXI_SHARE}] ${MODELPARAM_VALUE.AXI_SHARE}
}

proc update_MODELPARAM_VALUE.FREQ { MODELPARAM_VALUE.FREQ PARAM_VALUE.FREQ } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.FREQ}] ${MODELPARAM_VALUE.FREQ}
}

proc update_MODELPARAM_VALUE.C_ADDR_W { MODELPARAM_VALUE.C_ADDR_W PARAM_VALUE.C_ADDR_W } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.C_ADDR_W}] ${MODELPARAM_VALUE.C_ADDR_W}
}

proc update_MODELPARAM_VALUE.C_DATA_W { MODELPARAM_VALUE.C_DATA_W PARAM_VALUE.C_DATA_W } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.C_DATA_W}] ${MODELPARAM_VALUE.C_DATA_W}
}

proc update_MODELPARAM_VALUE.HP_DATA_BW { MODELPARAM_VALUE.HP_DATA_BW PARAM_VALUE.HP_DATA_BW } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.HP_DATA_BW}] ${MODELPARAM_VALUE.HP_DATA_BW}
}

proc update_MODELPARAM_VALUE.GP_DATA_BW { MODELPARAM_VALUE.GP_DATA_BW PARAM_VALUE.GP_DATA_BW } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.GP_DATA_BW}] ${MODELPARAM_VALUE.GP_DATA_BW}
}

proc update_MODELPARAM_VALUE.PP { MODELPARAM_VALUE.PP PARAM_VALUE.PP } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.PP}] ${MODELPARAM_VALUE.PP}
}

proc update_MODELPARAM_VALUE.CP { MODELPARAM_VALUE.CP PARAM_VALUE.CP } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.CP}] ${MODELPARAM_VALUE.CP}
}

proc update_MODELPARAM_VALUE.IBGRP_N { MODELPARAM_VALUE.IBGRP_N PARAM_VALUE.IBGRP_N } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.IBGRP_N}] ${MODELPARAM_VALUE.IBGRP_N}
}

proc update_MODELPARAM_VALUE.SEVN_DSP_CTL_0 { MODELPARAM_VALUE.SEVN_DSP_CTL_0 PARAM_VALUE.SEVN_DSP_CTL_0 } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.SEVN_DSP_CTL_0}] ${MODELPARAM_VALUE.SEVN_DSP_CTL_0}
}

proc update_MODELPARAM_VALUE.SEVN_DSP_CTL_1 { MODELPARAM_VALUE.SEVN_DSP_CTL_1 PARAM_VALUE.SEVN_DSP_CTL_1 } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.SEVN_DSP_CTL_1}] ${MODELPARAM_VALUE.SEVN_DSP_CTL_1}
}

proc update_MODELPARAM_VALUE.SEVN_DSP_CTL_2 { MODELPARAM_VALUE.SEVN_DSP_CTL_2 PARAM_VALUE.SEVN_DSP_CTL_2 } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.SEVN_DSP_CTL_2}] ${MODELPARAM_VALUE.SEVN_DSP_CTL_2}
}

proc update_MODELPARAM_VALUE.SEVN_DSP_CTL_3 { MODELPARAM_VALUE.SEVN_DSP_CTL_3 PARAM_VALUE.SEVN_DSP_CTL_3 } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.SEVN_DSP_CTL_3}] ${MODELPARAM_VALUE.SEVN_DSP_CTL_3}
}

proc update_MODELPARAM_VALUE.SEVN_DSP_CTL_4 { MODELPARAM_VALUE.SEVN_DSP_CTL_4 PARAM_VALUE.SEVN_DSP_CTL_4 } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.SEVN_DSP_CTL_4}] ${MODELPARAM_VALUE.SEVN_DSP_CTL_4}
}

proc update_MODELPARAM_VALUE.SEVN_DSP_CTL_5 { MODELPARAM_VALUE.SEVN_DSP_CTL_5 PARAM_VALUE.SEVN_DSP_CTL_5 } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.SEVN_DSP_CTL_5}] ${MODELPARAM_VALUE.SEVN_DSP_CTL_5}
}

proc update_MODELPARAM_VALUE.CONV_DSP_NUM { MODELPARAM_VALUE.CONV_DSP_NUM PARAM_VALUE.CONV_DSP_NUM } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.CONV_DSP_NUM}] ${MODELPARAM_VALUE.CONV_DSP_NUM}
}

proc update_MODELPARAM_VALUE.CONV_WFIFO_TYPE { MODELPARAM_VALUE.CONV_WFIFO_TYPE PARAM_VALUE.CONV_WFIFO_TYPE } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.CONV_WFIFO_TYPE}] ${MODELPARAM_VALUE.CONV_WFIFO_TYPE}
}

proc update_MODELPARAM_VALUE.NL_RATIO { MODELPARAM_VALUE.NL_RATIO PARAM_VALUE.NL_RATIO } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.NL_RATIO}] ${MODELPARAM_VALUE.NL_RATIO}
}

proc update_MODELPARAM_VALUE.NL_LEAKYRELU { MODELPARAM_VALUE.NL_LEAKYRELU PARAM_VALUE.NL_LEAKYRELU } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.NL_LEAKYRELU}] ${MODELPARAM_VALUE.NL_LEAKYRELU}
}

proc update_MODELPARAM_VALUE.MISC_WFIFO_TYPE { MODELPARAM_VALUE.MISC_WFIFO_TYPE PARAM_VALUE.MISC_WFIFO_TYPE } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.MISC_WFIFO_TYPE}] ${MODELPARAM_VALUE.MISC_WFIFO_TYPE}
}

proc update_MODELPARAM_VALUE.DWCONV_EN { MODELPARAM_VALUE.DWCONV_EN PARAM_VALUE.DWCONV_EN } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.DWCONV_EN}] ${MODELPARAM_VALUE.DWCONV_EN}
}

proc update_MODELPARAM_VALUE.MISC_PP_N { MODELPARAM_VALUE.MISC_PP_N PARAM_VALUE.MISC_PP_N } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.MISC_PP_N}] ${MODELPARAM_VALUE.MISC_PP_N}
}

proc update_MODELPARAM_VALUE.ENABLE_AVG_MODE { MODELPARAM_VALUE.ENABLE_AVG_MODE PARAM_VALUE.ENABLE_AVG_MODE } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.ENABLE_AVG_MODE}] ${MODELPARAM_VALUE.ENABLE_AVG_MODE}
}

proc update_MODELPARAM_VALUE.ENABLE_CH_OFFSET { MODELPARAM_VALUE.ENABLE_CH_OFFSET PARAM_VALUE.ENABLE_CH_OFFSET } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.ENABLE_CH_OFFSET}] ${MODELPARAM_VALUE.ENABLE_CH_OFFSET}
}

proc update_MODELPARAM_VALUE.DPU_WORK_CNT_EN { MODELPARAM_VALUE.DPU_WORK_CNT_EN PARAM_VALUE.DPU_WORK_CNT_EN } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.DPU_WORK_CNT_EN}] ${MODELPARAM_VALUE.DPU_WORK_CNT_EN}
}

proc update_MODELPARAM_VALUE.DPU_WORK_CNT { MODELPARAM_VALUE.DPU_WORK_CNT PARAM_VALUE.DPU_WORK_CNT } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.DPU_WORK_CNT}] ${MODELPARAM_VALUE.DPU_WORK_CNT}
}

