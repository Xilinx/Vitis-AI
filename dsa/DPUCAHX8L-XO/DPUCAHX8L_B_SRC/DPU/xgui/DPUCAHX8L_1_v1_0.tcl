# Definitional proc to organize widgets for parameters.
proc init_gui { IPINST } {
  ipgui::add_param $IPINST -name "Component_Name"
  #Adding Page
  set Page_0 [ipgui::add_page $IPINST -name "Page 0" -display_name {Configuration}]
  set_property tooltip {Configuration} ${Page_0}
  set DWC_EN [ipgui::add_param $IPINST -name "DWC_EN" -parent ${Page_0} -widget comboBox]
  set_property tooltip {Depthwise Convolution is only used in special Neural Networks, such as RetinaNet} ${DWC_EN}
  ipgui::add_param $IPINST -name "FREQ" -parent ${Page_0} -widget comboBox
  set LEAKYRELU [ipgui::add_param $IPINST -name "LEAKYRELU" -parent ${Page_0} -widget comboBox]
  set_property tooltip {Leaky-Relu is only used in special Neural Networks, such as Yolo} ${LEAKYRELU}


}

proc update_PARAM_VALUE.AUGMENTATION_H_PARALLELISM { PARAM_VALUE.AUGMENTATION_H_PARALLELISM } {
	# Procedure called to update AUGMENTATION_H_PARALLELISM when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.AUGMENTATION_H_PARALLELISM { PARAM_VALUE.AUGMENTATION_H_PARALLELISM } {
	# Procedure called to validate AUGMENTATION_H_PARALLELISM
	return true
}

proc update_PARAM_VALUE.CONV_DSP_NUM { PARAM_VALUE.CONV_DSP_NUM } {
	# Procedure called to update CONV_DSP_NUM when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.CONV_DSP_NUM { PARAM_VALUE.CONV_DSP_NUM } {
	# Procedure called to validate CONV_DSP_NUM
	return true
}

proc update_PARAM_VALUE.CROSS_DIE { PARAM_VALUE.CROSS_DIE } {
	# Procedure called to update CROSS_DIE when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.CROSS_DIE { PARAM_VALUE.CROSS_DIE } {
	# Procedure called to validate CROSS_DIE
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

proc update_PARAM_VALUE.DWC_EN { PARAM_VALUE.DWC_EN } {
	# Procedure called to update DWC_EN when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.DWC_EN { PARAM_VALUE.DWC_EN } {
	# Procedure called to validate DWC_EN
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

proc update_PARAM_VALUE.FREQ { PARAM_VALUE.FREQ } {
	# Procedure called to update FREQ when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.FREQ { PARAM_VALUE.FREQ } {
	# Procedure called to validate FREQ
	return true
}

proc update_PARAM_VALUE.LEAKYRELU { PARAM_VALUE.LEAKYRELU } {
	# Procedure called to update LEAKYRELU when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.LEAKYRELU { PARAM_VALUE.LEAKYRELU } {
	# Procedure called to validate LEAKYRELU
	return true
}

proc update_PARAM_VALUE.MAX_KER_W { PARAM_VALUE.MAX_KER_W } {
	# Procedure called to update MAX_KER_W when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.MAX_KER_W { PARAM_VALUE.MAX_KER_W } {
	# Procedure called to validate MAX_KER_W
	return true
}

proc update_PARAM_VALUE.MISC_PP_N { PARAM_VALUE.MISC_PP_N } {
	# Procedure called to update MISC_PP_N when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.MISC_PP_N { PARAM_VALUE.MISC_PP_N } {
	# Procedure called to validate MISC_PP_N
	return true
}

proc update_PARAM_VALUE.w { PARAM_VALUE.w } {
	# Procedure called to update w when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.w { PARAM_VALUE.w } {
	# Procedure called to validate w
	return true
}


proc update_MODELPARAM_VALUE.C_ADDR_W { MODELPARAM_VALUE.C_ADDR_W PARAM_VALUE.C_ADDR_W } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.C_ADDR_W}] ${MODELPARAM_VALUE.C_ADDR_W}
}

proc update_MODELPARAM_VALUE.C_DATA_W { MODELPARAM_VALUE.C_DATA_W PARAM_VALUE.C_DATA_W } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.C_DATA_W}] ${MODELPARAM_VALUE.C_DATA_W}
}

proc update_MODELPARAM_VALUE.FREQ { MODELPARAM_VALUE.FREQ PARAM_VALUE.FREQ } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.FREQ}] ${MODELPARAM_VALUE.FREQ}
}

proc update_MODELPARAM_VALUE.CROSS_DIE { MODELPARAM_VALUE.CROSS_DIE PARAM_VALUE.CROSS_DIE } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.CROSS_DIE}] ${MODELPARAM_VALUE.CROSS_DIE}
}

proc update_MODELPARAM_VALUE.DWC_EN { MODELPARAM_VALUE.DWC_EN PARAM_VALUE.DWC_EN } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.DWC_EN}] ${MODELPARAM_VALUE.DWC_EN}
}

proc update_MODELPARAM_VALUE.MAX_KER_W { MODELPARAM_VALUE.MAX_KER_W PARAM_VALUE.MAX_KER_W } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.MAX_KER_W}] ${MODELPARAM_VALUE.MAX_KER_W}
}

proc update_MODELPARAM_VALUE.LEAKYRELU { MODELPARAM_VALUE.LEAKYRELU PARAM_VALUE.LEAKYRELU } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.LEAKYRELU}] ${MODELPARAM_VALUE.LEAKYRELU}
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

proc update_MODELPARAM_VALUE.AUGMENTATION_H_PARALLELISM { MODELPARAM_VALUE.AUGMENTATION_H_PARALLELISM PARAM_VALUE.AUGMENTATION_H_PARALLELISM } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.AUGMENTATION_H_PARALLELISM}] ${MODELPARAM_VALUE.AUGMENTATION_H_PARALLELISM}
}

proc update_MODELPARAM_VALUE.CONV_DSP_NUM { MODELPARAM_VALUE.CONV_DSP_NUM PARAM_VALUE.CONV_DSP_NUM } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.CONV_DSP_NUM}] ${MODELPARAM_VALUE.CONV_DSP_NUM}
}

proc update_MODELPARAM_VALUE.w { MODELPARAM_VALUE.w PARAM_VALUE.w } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.w}] ${MODELPARAM_VALUE.w}
}

