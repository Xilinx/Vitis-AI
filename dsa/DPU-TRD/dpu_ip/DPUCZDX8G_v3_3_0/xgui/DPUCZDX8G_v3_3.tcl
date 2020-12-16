
# Loading additional proc with user specified bodies to compute parameter values.
source [file join [file dirname [file dirname [info script]]] gui/DPUCZDX8G_v3_3.gtcl]

# Definitional proc to organize widgets for parameters.
proc init_gui { IPINST } {
  ipgui::add_param $IPINST -name "Component_Name"
  #Adding Page
  set Arch [ipgui::add_page $IPINST -name "Arch"]
  ipgui::add_param $IPINST -name "VER_DPU_NUM" -parent ${Arch} -widget comboBox
  ipgui::add_param $IPINST -name "ARCH" -parent ${Arch} -widget comboBox
  set ARCH_IMG_BKGRP [ipgui::add_param $IPINST -name "ARCH_IMG_BKGRP" -parent ${Arch} -widget comboBox]
  set_property tooltip {Select whether more on-chip RAMs are used} ${ARCH_IMG_BKGRP}
  set LOAD_AUGM [ipgui::add_param $IPINST -name "LOAD_AUGM" -parent ${Arch} -widget comboBox]
  set_property tooltip {Enablement of Channel Augmentation} ${LOAD_AUGM}
  set DWCV_ENA [ipgui::add_param $IPINST -name "DWCV_ENA" -parent ${Arch} -widget comboBox]
  set_property tooltip {Enablement of DepthWiseConv} ${DWCV_ENA}
  set ELEW_MULT_EN [ipgui::add_param $IPINST -name "ELEW_MULT_EN" -parent ${Arch} -widget comboBox]
  set_property tooltip {Enablement of ElementWise Multiply} ${ELEW_MULT_EN}
  set POOL_AVERAGE [ipgui::add_param $IPINST -name "POOL_AVERAGE" -parent ${Arch} -widget comboBox]
  set_property tooltip {Enablement of AveragePool} ${POOL_AVERAGE}
  #Adding Group
  set CONV [ipgui::add_group $IPINST -name "CONV" -parent ${Arch} -display_name {Conv}]
  set CONV_RELU_ADDON [ipgui::add_param $IPINST -name "CONV_RELU_ADDON" -parent ${CONV} -widget comboBox]
  set_property tooltip {Select the ReLU Type of Conv.} ${CONV_RELU_ADDON}

  #Adding Group
  set SFM [ipgui::add_group $IPINST -name "SFM" -parent ${Arch} -display_name {Softmax}]
  ipgui::add_param $IPINST -name "SFM_ENA" -parent ${SFM} -widget comboBox


  #Adding Page
  set Advanced [ipgui::add_page $IPINST -name "Advanced"]
  ipgui::add_param $IPINST -name "S_AXI_CLK_INDEPENDENT" -parent ${Advanced} -layout horizontal
  #Adding Group
  set IMPL [ipgui::add_group $IPINST -name "IMPL" -parent ${Advanced} -display_name {Implementation}]
  set CLK_GATING_ENA [ipgui::add_param $IPINST -name "CLK_GATING_ENA" -parent ${IMPL} -widget comboBox]
  set_property tooltip {Select whether the clock gating is enabled by connecting CLK_CE signals to each BUFGCE_DIVs} ${CLK_GATING_ENA}
  set CONV_DSP_CASC_MAX [ipgui::add_param $IPINST -name "CONV_DSP_CASC_MAX" -parent ${IMPL}]
  set_property tooltip {Select maximal length of DSP48 slice cascade chain} ${CONV_DSP_CASC_MAX}
  set CONV_DSP_ACCU_ENA [ipgui::add_param $IPINST -name "CONV_DSP_ACCU_ENA" -parent ${IMPL} -widget comboBox]
  set_property tooltip {Select whether more DSP Slices are used} ${CONV_DSP_ACCU_ENA}
  set URAM_N_USER [ipgui::add_param $IPINST -name "URAM_N_USER" -parent ${IMPL} -show_range false]
  set_property tooltip {Select utilization of Ultra-RAM per DPU} ${URAM_N_USER}

  #Adding Group
  set TMSTP [ipgui::add_group $IPINST -name "TMSTP" -parent ${Advanced} -display_name {TIMESTAMP}]
  set TIMESTAMP_ENA [ipgui::add_param $IPINST -name "TIMESTAMP_ENA" -parent ${TMSTP} -widget comboBox]
  set_property tooltip {Select whether the timestamp will be auto-update} ${TIMESTAMP_ENA}


  #Adding Page
  set Summary [ipgui::add_page $IPINST -name "Summary"]
  ipgui::add_param $IPINST -name "SUM_VER_TARGET" -parent ${Summary}
  ipgui::add_param $IPINST -name "SUM_AXI_PROTOCOL" -parent ${Summary} -widget comboBox
  ipgui::add_param $IPINST -name "SUM_S_AXI_DATA_BW" -parent ${Summary}
  ipgui::add_param $IPINST -name "SUM_GP_DATA_BW" -parent ${Summary}
  ipgui::add_param $IPINST -name "SUM_HP_DATA_BW" -parent ${Summary}
  ipgui::add_param $IPINST -name "SUM_SFM_HP_DATA_BW" -parent ${Summary}
  ipgui::add_param $IPINST -name "GP_ID_BW" -parent ${Summary} -show_range false
  ipgui::add_param $IPINST -name "SUM_DSP_NUM" -parent ${Summary}
  ipgui::add_param $IPINST -name "SUM_URAM_N" -parent ${Summary}
  ipgui::add_param $IPINST -name "SUM_BRAM_N" -parent ${Summary}


}

proc update_PARAM_VALUE.ARCH_ICP { PARAM_VALUE.ARCH_ICP PARAM_VALUE.ARCH } {
	# Procedure called to update ARCH_ICP when any of the dependent parameters in the arguments change
	
	set ARCH_ICP ${PARAM_VALUE.ARCH_ICP}
	set ARCH ${PARAM_VALUE.ARCH}
	set values(ARCH) [get_property value $ARCH]
	set_property value [gen_USERPARAMETER_ARCH_ICP_VALUE $values(ARCH)] $ARCH_ICP
}

proc validate_PARAM_VALUE.ARCH_ICP { PARAM_VALUE.ARCH_ICP } {
	# Procedure called to validate ARCH_ICP
	return true
}

proc update_PARAM_VALUE.ARCH_OCP { PARAM_VALUE.ARCH_OCP PARAM_VALUE.ARCH_ICP } {
	# Procedure called to update ARCH_OCP when any of the dependent parameters in the arguments change
	
	set ARCH_OCP ${PARAM_VALUE.ARCH_OCP}
	set ARCH_ICP ${PARAM_VALUE.ARCH_ICP}
	set values(ARCH_ICP) [get_property value $ARCH_ICP]
	set_property value [gen_USERPARAMETER_ARCH_OCP_VALUE $values(ARCH_ICP)] $ARCH_OCP
}

proc validate_PARAM_VALUE.ARCH_OCP { PARAM_VALUE.ARCH_OCP } {
	# Procedure called to validate ARCH_OCP
	return true
}

proc update_PARAM_VALUE.ARCH_PP { PARAM_VALUE.ARCH_PP PARAM_VALUE.ARCH } {
	# Procedure called to update ARCH_PP when any of the dependent parameters in the arguments change
	
	set ARCH_PP ${PARAM_VALUE.ARCH_PP}
	set ARCH ${PARAM_VALUE.ARCH}
	set values(ARCH) [get_property value $ARCH]
	set_property value [gen_USERPARAMETER_ARCH_PP_VALUE $values(ARCH)] $ARCH_PP
}

proc validate_PARAM_VALUE.ARCH_PP { PARAM_VALUE.ARCH_PP } {
	# Procedure called to validate ARCH_PP
	return true
}

proc update_PARAM_VALUE.BANK_IMG_N { PARAM_VALUE.BANK_IMG_N PARAM_VALUE.ARCH_IMG_BKGRP PARAM_VALUE.ARCH_PP } {
	# Procedure called to update BANK_IMG_N when any of the dependent parameters in the arguments change
	
	set BANK_IMG_N ${PARAM_VALUE.BANK_IMG_N}
	set ARCH_IMG_BKGRP ${PARAM_VALUE.ARCH_IMG_BKGRP}
	set ARCH_PP ${PARAM_VALUE.ARCH_PP}
	set values(ARCH_IMG_BKGRP) [get_property value $ARCH_IMG_BKGRP]
	set values(ARCH_PP) [get_property value $ARCH_PP]
	set_property value [gen_USERPARAMETER_BANK_IMG_N_VALUE $values(ARCH_IMG_BKGRP) $values(ARCH_PP)] $BANK_IMG_N
}

proc validate_PARAM_VALUE.BANK_IMG_N { PARAM_VALUE.BANK_IMG_N } {
	# Procedure called to validate BANK_IMG_N
	return true
}

proc update_PARAM_VALUE.BANK_WGT_N { PARAM_VALUE.BANK_WGT_N PARAM_VALUE.ARCH_OCP PARAM_VALUE.DWCV_ENA } {
	# Procedure called to update BANK_WGT_N when any of the dependent parameters in the arguments change
	
	set BANK_WGT_N ${PARAM_VALUE.BANK_WGT_N}
	set ARCH_OCP ${PARAM_VALUE.ARCH_OCP}
	set DWCV_ENA ${PARAM_VALUE.DWCV_ENA}
	set values(ARCH_OCP) [get_property value $ARCH_OCP]
	set values(DWCV_ENA) [get_property value $DWCV_ENA]
	set_property value [gen_USERPARAMETER_BANK_WGT_N_VALUE $values(ARCH_OCP) $values(DWCV_ENA)] $BANK_WGT_N
}

proc validate_PARAM_VALUE.BANK_WGT_N { PARAM_VALUE.BANK_WGT_N } {
	# Procedure called to validate BANK_WGT_N
	return true
}

proc update_PARAM_VALUE.BBANK_BIAS { PARAM_VALUE.BBANK_BIAS PARAM_VALUE.BANK_BIAS PARAM_VALUE.UBANK_BIAS PARAM_VALUE.DBANK_BIAS } {
	# Procedure called to update BBANK_BIAS when any of the dependent parameters in the arguments change
	
	set BBANK_BIAS ${PARAM_VALUE.BBANK_BIAS}
	set BANK_BIAS ${PARAM_VALUE.BANK_BIAS}
	set UBANK_BIAS ${PARAM_VALUE.UBANK_BIAS}
	set DBANK_BIAS ${PARAM_VALUE.DBANK_BIAS}
	set values(BANK_BIAS) [get_property value $BANK_BIAS]
	set values(UBANK_BIAS) [get_property value $UBANK_BIAS]
	set values(DBANK_BIAS) [get_property value $DBANK_BIAS]
	set_property value [gen_USERPARAMETER_BBANK_BIAS_VALUE $values(BANK_BIAS) $values(UBANK_BIAS) $values(DBANK_BIAS)] $BBANK_BIAS
}

proc validate_PARAM_VALUE.BBANK_BIAS { PARAM_VALUE.BBANK_BIAS } {
	# Procedure called to validate BBANK_BIAS
	return true
}

proc update_PARAM_VALUE.BBANK_IMG_N { PARAM_VALUE.BBANK_IMG_N PARAM_VALUE.BANK_IMG_N PARAM_VALUE.UBANK_IMG_N PARAM_VALUE.DBANK_IMG_N } {
	# Procedure called to update BBANK_IMG_N when any of the dependent parameters in the arguments change
	
	set BBANK_IMG_N ${PARAM_VALUE.BBANK_IMG_N}
	set BANK_IMG_N ${PARAM_VALUE.BANK_IMG_N}
	set UBANK_IMG_N ${PARAM_VALUE.UBANK_IMG_N}
	set DBANK_IMG_N ${PARAM_VALUE.DBANK_IMG_N}
	set values(BANK_IMG_N) [get_property value $BANK_IMG_N]
	set values(UBANK_IMG_N) [get_property value $UBANK_IMG_N]
	set values(DBANK_IMG_N) [get_property value $DBANK_IMG_N]
	set_property value [gen_USERPARAMETER_BBANK_IMG_N_VALUE $values(BANK_IMG_N) $values(UBANK_IMG_N) $values(DBANK_IMG_N)] $BBANK_IMG_N
}

proc validate_PARAM_VALUE.BBANK_IMG_N { PARAM_VALUE.BBANK_IMG_N } {
	# Procedure called to validate BBANK_IMG_N
	return true
}

proc update_PARAM_VALUE.BBANK_WGT_N { PARAM_VALUE.BBANK_WGT_N PARAM_VALUE.BANK_WGT_N PARAM_VALUE.UBANK_WGT_N PARAM_VALUE.DBANK_WGT_N } {
	# Procedure called to update BBANK_WGT_N when any of the dependent parameters in the arguments change
	
	set BBANK_WGT_N ${PARAM_VALUE.BBANK_WGT_N}
	set BANK_WGT_N ${PARAM_VALUE.BANK_WGT_N}
	set UBANK_WGT_N ${PARAM_VALUE.UBANK_WGT_N}
	set DBANK_WGT_N ${PARAM_VALUE.DBANK_WGT_N}
	set values(BANK_WGT_N) [get_property value $BANK_WGT_N]
	set values(UBANK_WGT_N) [get_property value $UBANK_WGT_N]
	set values(DBANK_WGT_N) [get_property value $DBANK_WGT_N]
	set_property value [gen_USERPARAMETER_BBANK_WGT_N_VALUE $values(BANK_WGT_N) $values(UBANK_WGT_N) $values(DBANK_WGT_N)] $BBANK_WGT_N
}

proc validate_PARAM_VALUE.BBANK_WGT_N { PARAM_VALUE.BBANK_WGT_N } {
	# Procedure called to validate BBANK_WGT_N
	return true
}

proc update_PARAM_VALUE.CLK_GATING_ENA { PARAM_VALUE.CLK_GATING_ENA PARAM_VALUE.M_AXI_AWRLEN_BW } {
	# Procedure called to update CLK_GATING_ENA when any of the dependent parameters in the arguments change
	
	set CLK_GATING_ENA ${PARAM_VALUE.CLK_GATING_ENA}
	set M_AXI_AWRLEN_BW ${PARAM_VALUE.M_AXI_AWRLEN_BW}
	set values(M_AXI_AWRLEN_BW) [get_property value $M_AXI_AWRLEN_BW]
	if { [gen_USERPARAMETER_CLK_GATING_ENA_ENABLEMENT $values(M_AXI_AWRLEN_BW)] } {
		set_property enabled true $CLK_GATING_ENA
	} else {
		set_property enabled false $CLK_GATING_ENA
	}
}

proc validate_PARAM_VALUE.CLK_GATING_ENA { PARAM_VALUE.CLK_GATING_ENA } {
	# Procedure called to validate CLK_GATING_ENA
	return true
}

proc update_PARAM_VALUE.CONV_DSP_CASC_MAX { PARAM_VALUE.CONV_DSP_CASC_MAX PARAM_VALUE.M_AXI_AWRLEN_BW } {
	# Procedure called to update CONV_DSP_CASC_MAX when any of the dependent parameters in the arguments change
	
	set CONV_DSP_CASC_MAX ${PARAM_VALUE.CONV_DSP_CASC_MAX}
	set M_AXI_AWRLEN_BW ${PARAM_VALUE.M_AXI_AWRLEN_BW}
	set values(M_AXI_AWRLEN_BW) [get_property value $M_AXI_AWRLEN_BW]
	if { [gen_USERPARAMETER_CONV_DSP_CASC_MAX_ENABLEMENT $values(M_AXI_AWRLEN_BW)] } {
		set_property enabled true $CONV_DSP_CASC_MAX
	} else {
		set_property enabled false $CONV_DSP_CASC_MAX
	}
}

proc validate_PARAM_VALUE.CONV_DSP_CASC_MAX { PARAM_VALUE.CONV_DSP_CASC_MAX } {
	# Procedure called to validate CONV_DSP_CASC_MAX
	return true
}

proc update_PARAM_VALUE.CONV_DSP_NUM { PARAM_VALUE.CONV_DSP_NUM PARAM_VALUE.ARCH_PP PARAM_VALUE.ARCH_ICP PARAM_VALUE.ARCH_OCP PARAM_VALUE.CONV_DSP_ACCU_ENA } {
	# Procedure called to update CONV_DSP_NUM when any of the dependent parameters in the arguments change
	
	set CONV_DSP_NUM ${PARAM_VALUE.CONV_DSP_NUM}
	set ARCH_PP ${PARAM_VALUE.ARCH_PP}
	set ARCH_ICP ${PARAM_VALUE.ARCH_ICP}
	set ARCH_OCP ${PARAM_VALUE.ARCH_OCP}
	set CONV_DSP_ACCU_ENA ${PARAM_VALUE.CONV_DSP_ACCU_ENA}
	set values(ARCH_PP) [get_property value $ARCH_PP]
	set values(ARCH_ICP) [get_property value $ARCH_ICP]
	set values(ARCH_OCP) [get_property value $ARCH_OCP]
	set values(CONV_DSP_ACCU_ENA) [get_property value $CONV_DSP_ACCU_ENA]
	set_property value [gen_USERPARAMETER_CONV_DSP_NUM_VALUE $values(ARCH_PP) $values(ARCH_ICP) $values(ARCH_OCP) $values(CONV_DSP_ACCU_ENA)] $CONV_DSP_NUM
}

proc validate_PARAM_VALUE.CONV_DSP_NUM { PARAM_VALUE.CONV_DSP_NUM } {
	# Procedure called to validate CONV_DSP_NUM
	return true
}

proc update_PARAM_VALUE.CONV_LEAKYRELU { PARAM_VALUE.CONV_LEAKYRELU PARAM_VALUE.CONV_RELU_ADDON } {
	# Procedure called to update CONV_LEAKYRELU when any of the dependent parameters in the arguments change
	
	set CONV_LEAKYRELU ${PARAM_VALUE.CONV_LEAKYRELU}
	set CONV_RELU_ADDON ${PARAM_VALUE.CONV_RELU_ADDON}
	set values(CONV_RELU_ADDON) [get_property value $CONV_RELU_ADDON]
	set_property value [gen_USERPARAMETER_CONV_LEAKYRELU_VALUE $values(CONV_RELU_ADDON)] $CONV_LEAKYRELU
}

proc validate_PARAM_VALUE.CONV_LEAKYRELU { PARAM_VALUE.CONV_LEAKYRELU } {
	# Procedure called to validate CONV_LEAKYRELU
	return true
}

proc update_PARAM_VALUE.CONV_RELU6 { PARAM_VALUE.CONV_RELU6 PARAM_VALUE.CONV_RELU_ADDON } {
	# Procedure called to update CONV_RELU6 when any of the dependent parameters in the arguments change
	
	set CONV_RELU6 ${PARAM_VALUE.CONV_RELU6}
	set CONV_RELU_ADDON ${PARAM_VALUE.CONV_RELU_ADDON}
	set values(CONV_RELU_ADDON) [get_property value $CONV_RELU_ADDON]
	set_property value [gen_USERPARAMETER_CONV_RELU6_VALUE $values(CONV_RELU_ADDON)] $CONV_RELU6
}

proc validate_PARAM_VALUE.CONV_RELU6 { PARAM_VALUE.CONV_RELU6 } {
	# Procedure called to validate CONV_RELU6
	return true
}

proc update_PARAM_VALUE.DNNDK_PRINT { PARAM_VALUE.DNNDK_PRINT PARAM_VALUE.VER_DPU_NUM PARAM_VALUE.ARCH PARAM_VALUE.ARCH_IMG_BKGRP PARAM_VALUE.LOAD_AUGM PARAM_VALUE.DWCV_ENA PARAM_VALUE.POOL_AVERAGE PARAM_VALUE.CONV_RELU_ADDON PARAM_VALUE.SFM_ENA PARAM_VALUE.S_AXI_CLK_INDEPENDENT PARAM_VALUE.CLK_GATING_ENA PARAM_VALUE.CONV_DSP_CASC_MAX PARAM_VALUE.CONV_DSP_ACCU_ENA PARAM_VALUE.URAM_N_USER PARAM_VALUE.TIMESTAMP_ENA PARAM_VALUE.SUM_VER_TARGET PARAM_VALUE.SUM_AXI_PROTOCOL PARAM_VALUE.SUM_S_AXI_DATA_BW PARAM_VALUE.SUM_GP_DATA_BW PARAM_VALUE.SUM_HP_DATA_BW PARAM_VALUE.SUM_SFM_HP_DATA_BW PARAM_VALUE.GP_ID_BW PARAM_VALUE.SUM_DSP_NUM PARAM_VALUE.SUM_URAM_N PARAM_VALUE.SUM_BRAM_N } {
	# Procedure called to update DNNDK_PRINT when any of the dependent parameters in the arguments change
	
	set DNNDK_PRINT ${PARAM_VALUE.DNNDK_PRINT}
	set VER_DPU_NUM ${PARAM_VALUE.VER_DPU_NUM}
	set ARCH ${PARAM_VALUE.ARCH}
	set ARCH_IMG_BKGRP ${PARAM_VALUE.ARCH_IMG_BKGRP}
	set LOAD_AUGM ${PARAM_VALUE.LOAD_AUGM}
	set DWCV_ENA ${PARAM_VALUE.DWCV_ENA}
	set POOL_AVERAGE ${PARAM_VALUE.POOL_AVERAGE}
	set CONV_RELU_ADDON ${PARAM_VALUE.CONV_RELU_ADDON}
	set SFM_ENA ${PARAM_VALUE.SFM_ENA}
	set S_AXI_CLK_INDEPENDENT ${PARAM_VALUE.S_AXI_CLK_INDEPENDENT}
	set CLK_GATING_ENA ${PARAM_VALUE.CLK_GATING_ENA}
	set CONV_DSP_CASC_MAX ${PARAM_VALUE.CONV_DSP_CASC_MAX}
	set CONV_DSP_ACCU_ENA ${PARAM_VALUE.CONV_DSP_ACCU_ENA}
	set URAM_N_USER ${PARAM_VALUE.URAM_N_USER}
	set TIMESTAMP_ENA ${PARAM_VALUE.TIMESTAMP_ENA}
	set SUM_VER_TARGET ${PARAM_VALUE.SUM_VER_TARGET}
	set SUM_AXI_PROTOCOL ${PARAM_VALUE.SUM_AXI_PROTOCOL}
	set SUM_S_AXI_DATA_BW ${PARAM_VALUE.SUM_S_AXI_DATA_BW}
	set SUM_GP_DATA_BW ${PARAM_VALUE.SUM_GP_DATA_BW}
	set SUM_HP_DATA_BW ${PARAM_VALUE.SUM_HP_DATA_BW}
	set SUM_SFM_HP_DATA_BW ${PARAM_VALUE.SUM_SFM_HP_DATA_BW}
	set GP_ID_BW ${PARAM_VALUE.GP_ID_BW}
	set SUM_DSP_NUM ${PARAM_VALUE.SUM_DSP_NUM}
	set SUM_URAM_N ${PARAM_VALUE.SUM_URAM_N}
	set SUM_BRAM_N ${PARAM_VALUE.SUM_BRAM_N}
	set values(VER_DPU_NUM) [get_property value $VER_DPU_NUM]
	set values(ARCH) [get_property value $ARCH]
	set values(ARCH_IMG_BKGRP) [get_property value $ARCH_IMG_BKGRP]
	set values(LOAD_AUGM) [get_property value $LOAD_AUGM]
	set values(DWCV_ENA) [get_property value $DWCV_ENA]
	set values(POOL_AVERAGE) [get_property value $POOL_AVERAGE]
	set values(CONV_RELU_ADDON) [get_property value $CONV_RELU_ADDON]
	set values(SFM_ENA) [get_property value $SFM_ENA]
	set values(S_AXI_CLK_INDEPENDENT) [get_property value $S_AXI_CLK_INDEPENDENT]
	set values(CLK_GATING_ENA) [get_property value $CLK_GATING_ENA]
	set values(CONV_DSP_CASC_MAX) [get_property value $CONV_DSP_CASC_MAX]
	set values(CONV_DSP_ACCU_ENA) [get_property value $CONV_DSP_ACCU_ENA]
	set values(URAM_N_USER) [get_property value $URAM_N_USER]
	set values(TIMESTAMP_ENA) [get_property value $TIMESTAMP_ENA]
	set values(SUM_VER_TARGET) [get_property value $SUM_VER_TARGET]
	set values(SUM_AXI_PROTOCOL) [get_property value $SUM_AXI_PROTOCOL]
	set values(SUM_S_AXI_DATA_BW) [get_property value $SUM_S_AXI_DATA_BW]
	set values(SUM_GP_DATA_BW) [get_property value $SUM_GP_DATA_BW]
	set values(SUM_HP_DATA_BW) [get_property value $SUM_HP_DATA_BW]
	set values(SUM_SFM_HP_DATA_BW) [get_property value $SUM_SFM_HP_DATA_BW]
	set values(GP_ID_BW) [get_property value $GP_ID_BW]
	set values(SUM_DSP_NUM) [get_property value $SUM_DSP_NUM]
	set values(SUM_URAM_N) [get_property value $SUM_URAM_N]
	set values(SUM_BRAM_N) [get_property value $SUM_BRAM_N]
	set_property value [gen_USERPARAMETER_DNNDK_PRINT_VALUE $values(VER_DPU_NUM) $values(ARCH) $values(ARCH_IMG_BKGRP) $values(LOAD_AUGM) $values(DWCV_ENA) $values(POOL_AVERAGE) $values(CONV_RELU_ADDON) $values(SFM_ENA) $values(S_AXI_CLK_INDEPENDENT) $values(CLK_GATING_ENA) $values(CONV_DSP_CASC_MAX) $values(CONV_DSP_ACCU_ENA) $values(URAM_N_USER) $values(TIMESTAMP_ENA) $values(SUM_VER_TARGET) $values(SUM_AXI_PROTOCOL) $values(SUM_S_AXI_DATA_BW) $values(SUM_GP_DATA_BW) $values(SUM_HP_DATA_BW) $values(SUM_SFM_HP_DATA_BW) $values(GP_ID_BW) $values(SUM_DSP_NUM) $values(SUM_URAM_N) $values(SUM_BRAM_N)] $DNNDK_PRINT
}

proc validate_PARAM_VALUE.DNNDK_PRINT { PARAM_VALUE.DNNDK_PRINT } {
	# Procedure called to validate DNNDK_PRINT
	return true
}

proc update_PARAM_VALUE.DPU1_DBANK_BIAS { PARAM_VALUE.DPU1_DBANK_BIAS PARAM_VALUE.DBANK_BIAS } {
	# Procedure called to update DPU1_DBANK_BIAS when any of the dependent parameters in the arguments change
	
	set DPU1_DBANK_BIAS ${PARAM_VALUE.DPU1_DBANK_BIAS}
	set DBANK_BIAS ${PARAM_VALUE.DBANK_BIAS}
	set values(DBANK_BIAS) [get_property value $DBANK_BIAS]
	set_property value [gen_USERPARAMETER_DPU1_DBANK_BIAS_VALUE $values(DBANK_BIAS)] $DPU1_DBANK_BIAS
}

proc validate_PARAM_VALUE.DPU1_DBANK_BIAS { PARAM_VALUE.DPU1_DBANK_BIAS } {
	# Procedure called to validate DPU1_DBANK_BIAS
	return true
}

proc update_PARAM_VALUE.DPU1_DBANK_IMG_N { PARAM_VALUE.DPU1_DBANK_IMG_N PARAM_VALUE.DBANK_IMG_N } {
	# Procedure called to update DPU1_DBANK_IMG_N when any of the dependent parameters in the arguments change
	
	set DPU1_DBANK_IMG_N ${PARAM_VALUE.DPU1_DBANK_IMG_N}
	set DBANK_IMG_N ${PARAM_VALUE.DBANK_IMG_N}
	set values(DBANK_IMG_N) [get_property value $DBANK_IMG_N]
	set_property value [gen_USERPARAMETER_DPU1_DBANK_IMG_N_VALUE $values(DBANK_IMG_N)] $DPU1_DBANK_IMG_N
}

proc validate_PARAM_VALUE.DPU1_DBANK_IMG_N { PARAM_VALUE.DPU1_DBANK_IMG_N } {
	# Procedure called to validate DPU1_DBANK_IMG_N
	return true
}

proc update_PARAM_VALUE.DPU1_DBANK_WGT_N { PARAM_VALUE.DPU1_DBANK_WGT_N PARAM_VALUE.DBANK_WGT_N } {
	# Procedure called to update DPU1_DBANK_WGT_N when any of the dependent parameters in the arguments change
	
	set DPU1_DBANK_WGT_N ${PARAM_VALUE.DPU1_DBANK_WGT_N}
	set DBANK_WGT_N ${PARAM_VALUE.DBANK_WGT_N}
	set values(DBANK_WGT_N) [get_property value $DBANK_WGT_N]
	set_property value [gen_USERPARAMETER_DPU1_DBANK_WGT_N_VALUE $values(DBANK_WGT_N)] $DPU1_DBANK_WGT_N
}

proc validate_PARAM_VALUE.DPU1_DBANK_WGT_N { PARAM_VALUE.DPU1_DBANK_WGT_N } {
	# Procedure called to validate DPU1_DBANK_WGT_N
	return true
}

proc update_PARAM_VALUE.DPU1_GP_ID_BW { PARAM_VALUE.DPU1_GP_ID_BW PARAM_VALUE.GP_ID_BW } {
	# Procedure called to update DPU1_GP_ID_BW when any of the dependent parameters in the arguments change
	
	set DPU1_GP_ID_BW ${PARAM_VALUE.DPU1_GP_ID_BW}
	set GP_ID_BW ${PARAM_VALUE.GP_ID_BW}
	set values(GP_ID_BW) [get_property value $GP_ID_BW]
	set_property value [gen_USERPARAMETER_DPU1_GP_ID_BW_VALUE $values(GP_ID_BW)] $DPU1_GP_ID_BW
}

proc validate_PARAM_VALUE.DPU1_GP_ID_BW { PARAM_VALUE.DPU1_GP_ID_BW } {
	# Procedure called to validate DPU1_GP_ID_BW
	return true
}

proc update_PARAM_VALUE.DPU1_HP0_ID_BW { PARAM_VALUE.DPU1_HP0_ID_BW PARAM_VALUE.GP_ID_BW } {
	# Procedure called to update DPU1_HP0_ID_BW when any of the dependent parameters in the arguments change
	
	set DPU1_HP0_ID_BW ${PARAM_VALUE.DPU1_HP0_ID_BW}
	set GP_ID_BW ${PARAM_VALUE.GP_ID_BW}
	set values(GP_ID_BW) [get_property value $GP_ID_BW]
	set_property value [gen_USERPARAMETER_DPU1_HP0_ID_BW_VALUE $values(GP_ID_BW)] $DPU1_HP0_ID_BW
}

proc validate_PARAM_VALUE.DPU1_HP0_ID_BW { PARAM_VALUE.DPU1_HP0_ID_BW } {
	# Procedure called to validate DPU1_HP0_ID_BW
	return true
}

proc update_PARAM_VALUE.DPU1_HP1_ID_BW { PARAM_VALUE.DPU1_HP1_ID_BW PARAM_VALUE.GP_ID_BW } {
	# Procedure called to update DPU1_HP1_ID_BW when any of the dependent parameters in the arguments change
	
	set DPU1_HP1_ID_BW ${PARAM_VALUE.DPU1_HP1_ID_BW}
	set GP_ID_BW ${PARAM_VALUE.GP_ID_BW}
	set values(GP_ID_BW) [get_property value $GP_ID_BW]
	set_property value [gen_USERPARAMETER_DPU1_HP1_ID_BW_VALUE $values(GP_ID_BW)] $DPU1_HP1_ID_BW
}

proc validate_PARAM_VALUE.DPU1_HP1_ID_BW { PARAM_VALUE.DPU1_HP1_ID_BW } {
	# Procedure called to validate DPU1_HP1_ID_BW
	return true
}

proc update_PARAM_VALUE.DPU1_HP2_ID_BW { PARAM_VALUE.DPU1_HP2_ID_BW PARAM_VALUE.GP_ID_BW } {
	# Procedure called to update DPU1_HP2_ID_BW when any of the dependent parameters in the arguments change
	
	set DPU1_HP2_ID_BW ${PARAM_VALUE.DPU1_HP2_ID_BW}
	set GP_ID_BW ${PARAM_VALUE.GP_ID_BW}
	set values(GP_ID_BW) [get_property value $GP_ID_BW]
	set_property value [gen_USERPARAMETER_DPU1_HP2_ID_BW_VALUE $values(GP_ID_BW)] $DPU1_HP2_ID_BW
}

proc validate_PARAM_VALUE.DPU1_HP2_ID_BW { PARAM_VALUE.DPU1_HP2_ID_BW } {
	# Procedure called to validate DPU1_HP2_ID_BW
	return true
}

proc update_PARAM_VALUE.DPU1_HP3_ID_BW { PARAM_VALUE.DPU1_HP3_ID_BW PARAM_VALUE.GP_ID_BW } {
	# Procedure called to update DPU1_HP3_ID_BW when any of the dependent parameters in the arguments change
	
	set DPU1_HP3_ID_BW ${PARAM_VALUE.DPU1_HP3_ID_BW}
	set GP_ID_BW ${PARAM_VALUE.GP_ID_BW}
	set values(GP_ID_BW) [get_property value $GP_ID_BW]
	set_property value [gen_USERPARAMETER_DPU1_HP3_ID_BW_VALUE $values(GP_ID_BW)] $DPU1_HP3_ID_BW
}

proc validate_PARAM_VALUE.DPU1_HP3_ID_BW { PARAM_VALUE.DPU1_HP3_ID_BW } {
	# Procedure called to validate DPU1_HP3_ID_BW
	return true
}

proc update_PARAM_VALUE.DPU1_UBANK_BIAS { PARAM_VALUE.DPU1_UBANK_BIAS PARAM_VALUE.UBANK_BIAS_USER } {
	# Procedure called to update DPU1_UBANK_BIAS when any of the dependent parameters in the arguments change
	
	set DPU1_UBANK_BIAS ${PARAM_VALUE.DPU1_UBANK_BIAS}
	set UBANK_BIAS_USER ${PARAM_VALUE.UBANK_BIAS_USER}
	set values(UBANK_BIAS_USER) [get_property value $UBANK_BIAS_USER]
	set_property value [gen_USERPARAMETER_DPU1_UBANK_BIAS_VALUE $values(UBANK_BIAS_USER)] $DPU1_UBANK_BIAS
}

proc validate_PARAM_VALUE.DPU1_UBANK_BIAS { PARAM_VALUE.DPU1_UBANK_BIAS } {
	# Procedure called to validate DPU1_UBANK_BIAS
	return true
}

proc update_PARAM_VALUE.DPU1_UBANK_IMG_N { PARAM_VALUE.DPU1_UBANK_IMG_N PARAM_VALUE.UBANK_IMG_N_USER } {
	# Procedure called to update DPU1_UBANK_IMG_N when any of the dependent parameters in the arguments change
	
	set DPU1_UBANK_IMG_N ${PARAM_VALUE.DPU1_UBANK_IMG_N}
	set UBANK_IMG_N_USER ${PARAM_VALUE.UBANK_IMG_N_USER}
	set values(UBANK_IMG_N_USER) [get_property value $UBANK_IMG_N_USER]
	set_property value [gen_USERPARAMETER_DPU1_UBANK_IMG_N_VALUE $values(UBANK_IMG_N_USER)] $DPU1_UBANK_IMG_N
}

proc validate_PARAM_VALUE.DPU1_UBANK_IMG_N { PARAM_VALUE.DPU1_UBANK_IMG_N } {
	# Procedure called to validate DPU1_UBANK_IMG_N
	return true
}

proc update_PARAM_VALUE.DPU1_UBANK_WGT_N { PARAM_VALUE.DPU1_UBANK_WGT_N PARAM_VALUE.UBANK_WGT_N_USER } {
	# Procedure called to update DPU1_UBANK_WGT_N when any of the dependent parameters in the arguments change
	
	set DPU1_UBANK_WGT_N ${PARAM_VALUE.DPU1_UBANK_WGT_N}
	set UBANK_WGT_N_USER ${PARAM_VALUE.UBANK_WGT_N_USER}
	set values(UBANK_WGT_N_USER) [get_property value $UBANK_WGT_N_USER]
	set_property value [gen_USERPARAMETER_DPU1_UBANK_WGT_N_VALUE $values(UBANK_WGT_N_USER)] $DPU1_UBANK_WGT_N
}

proc validate_PARAM_VALUE.DPU1_UBANK_WGT_N { PARAM_VALUE.DPU1_UBANK_WGT_N } {
	# Procedure called to validate DPU1_UBANK_WGT_N
	return true
}

proc update_PARAM_VALUE.DPU2_DBANK_BIAS { PARAM_VALUE.DPU2_DBANK_BIAS PARAM_VALUE.DBANK_BIAS } {
	# Procedure called to update DPU2_DBANK_BIAS when any of the dependent parameters in the arguments change
	
	set DPU2_DBANK_BIAS ${PARAM_VALUE.DPU2_DBANK_BIAS}
	set DBANK_BIAS ${PARAM_VALUE.DBANK_BIAS}
	set values(DBANK_BIAS) [get_property value $DBANK_BIAS]
	set_property value [gen_USERPARAMETER_DPU2_DBANK_BIAS_VALUE $values(DBANK_BIAS)] $DPU2_DBANK_BIAS
}

proc validate_PARAM_VALUE.DPU2_DBANK_BIAS { PARAM_VALUE.DPU2_DBANK_BIAS } {
	# Procedure called to validate DPU2_DBANK_BIAS
	return true
}

proc update_PARAM_VALUE.DPU2_DBANK_IMG_N { PARAM_VALUE.DPU2_DBANK_IMG_N PARAM_VALUE.DBANK_IMG_N } {
	# Procedure called to update DPU2_DBANK_IMG_N when any of the dependent parameters in the arguments change
	
	set DPU2_DBANK_IMG_N ${PARAM_VALUE.DPU2_DBANK_IMG_N}
	set DBANK_IMG_N ${PARAM_VALUE.DBANK_IMG_N}
	set values(DBANK_IMG_N) [get_property value $DBANK_IMG_N]
	set_property value [gen_USERPARAMETER_DPU2_DBANK_IMG_N_VALUE $values(DBANK_IMG_N)] $DPU2_DBANK_IMG_N
}

proc validate_PARAM_VALUE.DPU2_DBANK_IMG_N { PARAM_VALUE.DPU2_DBANK_IMG_N } {
	# Procedure called to validate DPU2_DBANK_IMG_N
	return true
}

proc update_PARAM_VALUE.DPU2_DBANK_WGT_N { PARAM_VALUE.DPU2_DBANK_WGT_N PARAM_VALUE.DBANK_WGT_N } {
	# Procedure called to update DPU2_DBANK_WGT_N when any of the dependent parameters in the arguments change
	
	set DPU2_DBANK_WGT_N ${PARAM_VALUE.DPU2_DBANK_WGT_N}
	set DBANK_WGT_N ${PARAM_VALUE.DBANK_WGT_N}
	set values(DBANK_WGT_N) [get_property value $DBANK_WGT_N]
	set_property value [gen_USERPARAMETER_DPU2_DBANK_WGT_N_VALUE $values(DBANK_WGT_N)] $DPU2_DBANK_WGT_N
}

proc validate_PARAM_VALUE.DPU2_DBANK_WGT_N { PARAM_VALUE.DPU2_DBANK_WGT_N } {
	# Procedure called to validate DPU2_DBANK_WGT_N
	return true
}

proc update_PARAM_VALUE.DPU2_GP_ID_BW { PARAM_VALUE.DPU2_GP_ID_BW PARAM_VALUE.GP_ID_BW } {
	# Procedure called to update DPU2_GP_ID_BW when any of the dependent parameters in the arguments change
	
	set DPU2_GP_ID_BW ${PARAM_VALUE.DPU2_GP_ID_BW}
	set GP_ID_BW ${PARAM_VALUE.GP_ID_BW}
	set values(GP_ID_BW) [get_property value $GP_ID_BW]
	set_property value [gen_USERPARAMETER_DPU2_GP_ID_BW_VALUE $values(GP_ID_BW)] $DPU2_GP_ID_BW
}

proc validate_PARAM_VALUE.DPU2_GP_ID_BW { PARAM_VALUE.DPU2_GP_ID_BW } {
	# Procedure called to validate DPU2_GP_ID_BW
	return true
}

proc update_PARAM_VALUE.DPU2_HP0_ID_BW { PARAM_VALUE.DPU2_HP0_ID_BW PARAM_VALUE.GP_ID_BW } {
	# Procedure called to update DPU2_HP0_ID_BW when any of the dependent parameters in the arguments change
	
	set DPU2_HP0_ID_BW ${PARAM_VALUE.DPU2_HP0_ID_BW}
	set GP_ID_BW ${PARAM_VALUE.GP_ID_BW}
	set values(GP_ID_BW) [get_property value $GP_ID_BW]
	set_property value [gen_USERPARAMETER_DPU2_HP0_ID_BW_VALUE $values(GP_ID_BW)] $DPU2_HP0_ID_BW
}

proc validate_PARAM_VALUE.DPU2_HP0_ID_BW { PARAM_VALUE.DPU2_HP0_ID_BW } {
	# Procedure called to validate DPU2_HP0_ID_BW
	return true
}

proc update_PARAM_VALUE.DPU2_HP1_ID_BW { PARAM_VALUE.DPU2_HP1_ID_BW PARAM_VALUE.GP_ID_BW } {
	# Procedure called to update DPU2_HP1_ID_BW when any of the dependent parameters in the arguments change
	
	set DPU2_HP1_ID_BW ${PARAM_VALUE.DPU2_HP1_ID_BW}
	set GP_ID_BW ${PARAM_VALUE.GP_ID_BW}
	set values(GP_ID_BW) [get_property value $GP_ID_BW]
	set_property value [gen_USERPARAMETER_DPU2_HP1_ID_BW_VALUE $values(GP_ID_BW)] $DPU2_HP1_ID_BW
}

proc validate_PARAM_VALUE.DPU2_HP1_ID_BW { PARAM_VALUE.DPU2_HP1_ID_BW } {
	# Procedure called to validate DPU2_HP1_ID_BW
	return true
}

proc update_PARAM_VALUE.DPU2_HP2_ID_BW { PARAM_VALUE.DPU2_HP2_ID_BW PARAM_VALUE.GP_ID_BW } {
	# Procedure called to update DPU2_HP2_ID_BW when any of the dependent parameters in the arguments change
	
	set DPU2_HP2_ID_BW ${PARAM_VALUE.DPU2_HP2_ID_BW}
	set GP_ID_BW ${PARAM_VALUE.GP_ID_BW}
	set values(GP_ID_BW) [get_property value $GP_ID_BW]
	set_property value [gen_USERPARAMETER_DPU2_HP2_ID_BW_VALUE $values(GP_ID_BW)] $DPU2_HP2_ID_BW
}

proc validate_PARAM_VALUE.DPU2_HP2_ID_BW { PARAM_VALUE.DPU2_HP2_ID_BW } {
	# Procedure called to validate DPU2_HP2_ID_BW
	return true
}

proc update_PARAM_VALUE.DPU2_HP3_ID_BW { PARAM_VALUE.DPU2_HP3_ID_BW PARAM_VALUE.GP_ID_BW } {
	# Procedure called to update DPU2_HP3_ID_BW when any of the dependent parameters in the arguments change
	
	set DPU2_HP3_ID_BW ${PARAM_VALUE.DPU2_HP3_ID_BW}
	set GP_ID_BW ${PARAM_VALUE.GP_ID_BW}
	set values(GP_ID_BW) [get_property value $GP_ID_BW]
	set_property value [gen_USERPARAMETER_DPU2_HP3_ID_BW_VALUE $values(GP_ID_BW)] $DPU2_HP3_ID_BW
}

proc validate_PARAM_VALUE.DPU2_HP3_ID_BW { PARAM_VALUE.DPU2_HP3_ID_BW } {
	# Procedure called to validate DPU2_HP3_ID_BW
	return true
}

proc update_PARAM_VALUE.DPU2_UBANK_BIAS { PARAM_VALUE.DPU2_UBANK_BIAS PARAM_VALUE.UBANK_BIAS_USER } {
	# Procedure called to update DPU2_UBANK_BIAS when any of the dependent parameters in the arguments change
	
	set DPU2_UBANK_BIAS ${PARAM_VALUE.DPU2_UBANK_BIAS}
	set UBANK_BIAS_USER ${PARAM_VALUE.UBANK_BIAS_USER}
	set values(UBANK_BIAS_USER) [get_property value $UBANK_BIAS_USER]
	set_property value [gen_USERPARAMETER_DPU2_UBANK_BIAS_VALUE $values(UBANK_BIAS_USER)] $DPU2_UBANK_BIAS
}

proc validate_PARAM_VALUE.DPU2_UBANK_BIAS { PARAM_VALUE.DPU2_UBANK_BIAS } {
	# Procedure called to validate DPU2_UBANK_BIAS
	return true
}

proc update_PARAM_VALUE.DPU2_UBANK_IMG_N { PARAM_VALUE.DPU2_UBANK_IMG_N PARAM_VALUE.UBANK_IMG_N_USER } {
	# Procedure called to update DPU2_UBANK_IMG_N when any of the dependent parameters in the arguments change
	
	set DPU2_UBANK_IMG_N ${PARAM_VALUE.DPU2_UBANK_IMG_N}
	set UBANK_IMG_N_USER ${PARAM_VALUE.UBANK_IMG_N_USER}
	set values(UBANK_IMG_N_USER) [get_property value $UBANK_IMG_N_USER]
	set_property value [gen_USERPARAMETER_DPU2_UBANK_IMG_N_VALUE $values(UBANK_IMG_N_USER)] $DPU2_UBANK_IMG_N
}

proc validate_PARAM_VALUE.DPU2_UBANK_IMG_N { PARAM_VALUE.DPU2_UBANK_IMG_N } {
	# Procedure called to validate DPU2_UBANK_IMG_N
	return true
}

proc update_PARAM_VALUE.DPU2_UBANK_WGT_N { PARAM_VALUE.DPU2_UBANK_WGT_N PARAM_VALUE.UBANK_WGT_N_USER } {
	# Procedure called to update DPU2_UBANK_WGT_N when any of the dependent parameters in the arguments change
	
	set DPU2_UBANK_WGT_N ${PARAM_VALUE.DPU2_UBANK_WGT_N}
	set UBANK_WGT_N_USER ${PARAM_VALUE.UBANK_WGT_N_USER}
	set values(UBANK_WGT_N_USER) [get_property value $UBANK_WGT_N_USER]
	set_property value [gen_USERPARAMETER_DPU2_UBANK_WGT_N_VALUE $values(UBANK_WGT_N_USER)] $DPU2_UBANK_WGT_N
}

proc validate_PARAM_VALUE.DPU2_UBANK_WGT_N { PARAM_VALUE.DPU2_UBANK_WGT_N } {
	# Procedure called to validate DPU2_UBANK_WGT_N
	return true
}

proc update_PARAM_VALUE.DPU3_DBANK_BIAS { PARAM_VALUE.DPU3_DBANK_BIAS PARAM_VALUE.DBANK_BIAS } {
	# Procedure called to update DPU3_DBANK_BIAS when any of the dependent parameters in the arguments change
	
	set DPU3_DBANK_BIAS ${PARAM_VALUE.DPU3_DBANK_BIAS}
	set DBANK_BIAS ${PARAM_VALUE.DBANK_BIAS}
	set values(DBANK_BIAS) [get_property value $DBANK_BIAS]
	set_property value [gen_USERPARAMETER_DPU3_DBANK_BIAS_VALUE $values(DBANK_BIAS)] $DPU3_DBANK_BIAS
}

proc validate_PARAM_VALUE.DPU3_DBANK_BIAS { PARAM_VALUE.DPU3_DBANK_BIAS } {
	# Procedure called to validate DPU3_DBANK_BIAS
	return true
}

proc update_PARAM_VALUE.DPU3_DBANK_IMG_N { PARAM_VALUE.DPU3_DBANK_IMG_N PARAM_VALUE.DBANK_IMG_N } {
	# Procedure called to update DPU3_DBANK_IMG_N when any of the dependent parameters in the arguments change
	
	set DPU3_DBANK_IMG_N ${PARAM_VALUE.DPU3_DBANK_IMG_N}
	set DBANK_IMG_N ${PARAM_VALUE.DBANK_IMG_N}
	set values(DBANK_IMG_N) [get_property value $DBANK_IMG_N]
	set_property value [gen_USERPARAMETER_DPU3_DBANK_IMG_N_VALUE $values(DBANK_IMG_N)] $DPU3_DBANK_IMG_N
}

proc validate_PARAM_VALUE.DPU3_DBANK_IMG_N { PARAM_VALUE.DPU3_DBANK_IMG_N } {
	# Procedure called to validate DPU3_DBANK_IMG_N
	return true
}

proc update_PARAM_VALUE.DPU3_DBANK_WGT_N { PARAM_VALUE.DPU3_DBANK_WGT_N PARAM_VALUE.DBANK_WGT_N } {
	# Procedure called to update DPU3_DBANK_WGT_N when any of the dependent parameters in the arguments change
	
	set DPU3_DBANK_WGT_N ${PARAM_VALUE.DPU3_DBANK_WGT_N}
	set DBANK_WGT_N ${PARAM_VALUE.DBANK_WGT_N}
	set values(DBANK_WGT_N) [get_property value $DBANK_WGT_N]
	set_property value [gen_USERPARAMETER_DPU3_DBANK_WGT_N_VALUE $values(DBANK_WGT_N)] $DPU3_DBANK_WGT_N
}

proc validate_PARAM_VALUE.DPU3_DBANK_WGT_N { PARAM_VALUE.DPU3_DBANK_WGT_N } {
	# Procedure called to validate DPU3_DBANK_WGT_N
	return true
}

proc update_PARAM_VALUE.DPU3_GP_ID_BW { PARAM_VALUE.DPU3_GP_ID_BW PARAM_VALUE.GP_ID_BW } {
	# Procedure called to update DPU3_GP_ID_BW when any of the dependent parameters in the arguments change
	
	set DPU3_GP_ID_BW ${PARAM_VALUE.DPU3_GP_ID_BW}
	set GP_ID_BW ${PARAM_VALUE.GP_ID_BW}
	set values(GP_ID_BW) [get_property value $GP_ID_BW]
	set_property value [gen_USERPARAMETER_DPU3_GP_ID_BW_VALUE $values(GP_ID_BW)] $DPU3_GP_ID_BW
}

proc validate_PARAM_VALUE.DPU3_GP_ID_BW { PARAM_VALUE.DPU3_GP_ID_BW } {
	# Procedure called to validate DPU3_GP_ID_BW
	return true
}

proc update_PARAM_VALUE.DPU3_HP0_ID_BW { PARAM_VALUE.DPU3_HP0_ID_BW PARAM_VALUE.GP_ID_BW } {
	# Procedure called to update DPU3_HP0_ID_BW when any of the dependent parameters in the arguments change
	
	set DPU3_HP0_ID_BW ${PARAM_VALUE.DPU3_HP0_ID_BW}
	set GP_ID_BW ${PARAM_VALUE.GP_ID_BW}
	set values(GP_ID_BW) [get_property value $GP_ID_BW]
	set_property value [gen_USERPARAMETER_DPU3_HP0_ID_BW_VALUE $values(GP_ID_BW)] $DPU3_HP0_ID_BW
}

proc validate_PARAM_VALUE.DPU3_HP0_ID_BW { PARAM_VALUE.DPU3_HP0_ID_BW } {
	# Procedure called to validate DPU3_HP0_ID_BW
	return true
}

proc update_PARAM_VALUE.DPU3_HP1_ID_BW { PARAM_VALUE.DPU3_HP1_ID_BW PARAM_VALUE.GP_ID_BW } {
	# Procedure called to update DPU3_HP1_ID_BW when any of the dependent parameters in the arguments change
	
	set DPU3_HP1_ID_BW ${PARAM_VALUE.DPU3_HP1_ID_BW}
	set GP_ID_BW ${PARAM_VALUE.GP_ID_BW}
	set values(GP_ID_BW) [get_property value $GP_ID_BW]
	set_property value [gen_USERPARAMETER_DPU3_HP1_ID_BW_VALUE $values(GP_ID_BW)] $DPU3_HP1_ID_BW
}

proc validate_PARAM_VALUE.DPU3_HP1_ID_BW { PARAM_VALUE.DPU3_HP1_ID_BW } {
	# Procedure called to validate DPU3_HP1_ID_BW
	return true
}

proc update_PARAM_VALUE.DPU3_HP2_ID_BW { PARAM_VALUE.DPU3_HP2_ID_BW PARAM_VALUE.GP_ID_BW } {
	# Procedure called to update DPU3_HP2_ID_BW when any of the dependent parameters in the arguments change
	
	set DPU3_HP2_ID_BW ${PARAM_VALUE.DPU3_HP2_ID_BW}
	set GP_ID_BW ${PARAM_VALUE.GP_ID_BW}
	set values(GP_ID_BW) [get_property value $GP_ID_BW]
	set_property value [gen_USERPARAMETER_DPU3_HP2_ID_BW_VALUE $values(GP_ID_BW)] $DPU3_HP2_ID_BW
}

proc validate_PARAM_VALUE.DPU3_HP2_ID_BW { PARAM_VALUE.DPU3_HP2_ID_BW } {
	# Procedure called to validate DPU3_HP2_ID_BW
	return true
}

proc update_PARAM_VALUE.DPU3_HP3_ID_BW { PARAM_VALUE.DPU3_HP3_ID_BW PARAM_VALUE.GP_ID_BW } {
	# Procedure called to update DPU3_HP3_ID_BW when any of the dependent parameters in the arguments change
	
	set DPU3_HP3_ID_BW ${PARAM_VALUE.DPU3_HP3_ID_BW}
	set GP_ID_BW ${PARAM_VALUE.GP_ID_BW}
	set values(GP_ID_BW) [get_property value $GP_ID_BW]
	set_property value [gen_USERPARAMETER_DPU3_HP3_ID_BW_VALUE $values(GP_ID_BW)] $DPU3_HP3_ID_BW
}

proc validate_PARAM_VALUE.DPU3_HP3_ID_BW { PARAM_VALUE.DPU3_HP3_ID_BW } {
	# Procedure called to validate DPU3_HP3_ID_BW
	return true
}

proc update_PARAM_VALUE.DPU3_UBANK_BIAS { PARAM_VALUE.DPU3_UBANK_BIAS PARAM_VALUE.UBANK_BIAS_USER } {
	# Procedure called to update DPU3_UBANK_BIAS when any of the dependent parameters in the arguments change
	
	set DPU3_UBANK_BIAS ${PARAM_VALUE.DPU3_UBANK_BIAS}
	set UBANK_BIAS_USER ${PARAM_VALUE.UBANK_BIAS_USER}
	set values(UBANK_BIAS_USER) [get_property value $UBANK_BIAS_USER]
	set_property value [gen_USERPARAMETER_DPU3_UBANK_BIAS_VALUE $values(UBANK_BIAS_USER)] $DPU3_UBANK_BIAS
}

proc validate_PARAM_VALUE.DPU3_UBANK_BIAS { PARAM_VALUE.DPU3_UBANK_BIAS } {
	# Procedure called to validate DPU3_UBANK_BIAS
	return true
}

proc update_PARAM_VALUE.DPU3_UBANK_IMG_N { PARAM_VALUE.DPU3_UBANK_IMG_N PARAM_VALUE.UBANK_IMG_N_USER } {
	# Procedure called to update DPU3_UBANK_IMG_N when any of the dependent parameters in the arguments change
	
	set DPU3_UBANK_IMG_N ${PARAM_VALUE.DPU3_UBANK_IMG_N}
	set UBANK_IMG_N_USER ${PARAM_VALUE.UBANK_IMG_N_USER}
	set values(UBANK_IMG_N_USER) [get_property value $UBANK_IMG_N_USER]
	set_property value [gen_USERPARAMETER_DPU3_UBANK_IMG_N_VALUE $values(UBANK_IMG_N_USER)] $DPU3_UBANK_IMG_N
}

proc validate_PARAM_VALUE.DPU3_UBANK_IMG_N { PARAM_VALUE.DPU3_UBANK_IMG_N } {
	# Procedure called to validate DPU3_UBANK_IMG_N
	return true
}

proc update_PARAM_VALUE.DPU3_UBANK_WGT_N { PARAM_VALUE.DPU3_UBANK_WGT_N PARAM_VALUE.UBANK_WGT_N_USER } {
	# Procedure called to update DPU3_UBANK_WGT_N when any of the dependent parameters in the arguments change
	
	set DPU3_UBANK_WGT_N ${PARAM_VALUE.DPU3_UBANK_WGT_N}
	set UBANK_WGT_N_USER ${PARAM_VALUE.UBANK_WGT_N_USER}
	set values(UBANK_WGT_N_USER) [get_property value $UBANK_WGT_N_USER]
	set_property value [gen_USERPARAMETER_DPU3_UBANK_WGT_N_VALUE $values(UBANK_WGT_N_USER)] $DPU3_UBANK_WGT_N
}

proc validate_PARAM_VALUE.DPU3_UBANK_WGT_N { PARAM_VALUE.DPU3_UBANK_WGT_N } {
	# Procedure called to validate DPU3_UBANK_WGT_N
	return true
}

proc update_PARAM_VALUE.DWCV_DSP_NUM { PARAM_VALUE.DWCV_DSP_NUM PARAM_VALUE.DWCV_PARALLEL PARAM_VALUE.ARCH_ICP PARAM_VALUE.DWCV_ENA } {
	# Procedure called to update DWCV_DSP_NUM when any of the dependent parameters in the arguments change
	
	set DWCV_DSP_NUM ${PARAM_VALUE.DWCV_DSP_NUM}
	set DWCV_PARALLEL ${PARAM_VALUE.DWCV_PARALLEL}
	set ARCH_ICP ${PARAM_VALUE.ARCH_ICP}
	set DWCV_ENA ${PARAM_VALUE.DWCV_ENA}
	set values(DWCV_PARALLEL) [get_property value $DWCV_PARALLEL]
	set values(ARCH_ICP) [get_property value $ARCH_ICP]
	set values(DWCV_ENA) [get_property value $DWCV_ENA]
	set_property value [gen_USERPARAMETER_DWCV_DSP_NUM_VALUE $values(DWCV_PARALLEL) $values(ARCH_ICP) $values(DWCV_ENA)] $DWCV_DSP_NUM
}

proc validate_PARAM_VALUE.DWCV_DSP_NUM { PARAM_VALUE.DWCV_DSP_NUM } {
	# Procedure called to validate DWCV_DSP_NUM
	return true
}

proc update_PARAM_VALUE.DWCV_PARALLEL { PARAM_VALUE.DWCV_PARALLEL PARAM_VALUE.DWCV_ENA PARAM_VALUE.ARCH_PP } {
	# Procedure called to update DWCV_PARALLEL when any of the dependent parameters in the arguments change
	
	set DWCV_PARALLEL ${PARAM_VALUE.DWCV_PARALLEL}
	set DWCV_ENA ${PARAM_VALUE.DWCV_ENA}
	set ARCH_PP ${PARAM_VALUE.ARCH_PP}
	set values(DWCV_ENA) [get_property value $DWCV_ENA]
	set values(ARCH_PP) [get_property value $ARCH_PP]
	set_property value [gen_USERPARAMETER_DWCV_PARALLEL_VALUE $values(DWCV_ENA) $values(ARCH_PP)] $DWCV_PARALLEL
}

proc validate_PARAM_VALUE.DWCV_PARALLEL { PARAM_VALUE.DWCV_PARALLEL } {
	# Procedure called to validate DWCV_PARALLEL
	return true
}

proc update_PARAM_VALUE.DWCV_RELU6 { PARAM_VALUE.DWCV_RELU6 PARAM_VALUE.DWCV_ENA PARAM_VALUE.CONV_RELU6 } {
	# Procedure called to update DWCV_RELU6 when any of the dependent parameters in the arguments change
	
	set DWCV_RELU6 ${PARAM_VALUE.DWCV_RELU6}
	set DWCV_ENA ${PARAM_VALUE.DWCV_ENA}
	set CONV_RELU6 ${PARAM_VALUE.CONV_RELU6}
	set values(DWCV_ENA) [get_property value $DWCV_ENA]
	set values(CONV_RELU6) [get_property value $CONV_RELU6]
	set_property value [gen_USERPARAMETER_DWCV_RELU6_VALUE $values(DWCV_ENA) $values(CONV_RELU6)] $DWCV_RELU6
}

proc validate_PARAM_VALUE.DWCV_RELU6 { PARAM_VALUE.DWCV_RELU6 } {
	# Procedure called to validate DWCV_RELU6
	return true
}

proc update_PARAM_VALUE.HP0_ID_BW { PARAM_VALUE.HP0_ID_BW PARAM_VALUE.GP_ID_BW } {
	# Procedure called to update HP0_ID_BW when any of the dependent parameters in the arguments change
	
	set HP0_ID_BW ${PARAM_VALUE.HP0_ID_BW}
	set GP_ID_BW ${PARAM_VALUE.GP_ID_BW}
	set values(GP_ID_BW) [get_property value $GP_ID_BW]
	set_property value [gen_USERPARAMETER_HP0_ID_BW_VALUE $values(GP_ID_BW)] $HP0_ID_BW
}

proc validate_PARAM_VALUE.HP0_ID_BW { PARAM_VALUE.HP0_ID_BW } {
	# Procedure called to validate HP0_ID_BW
	return true
}

proc update_PARAM_VALUE.HP1_ID_BW { PARAM_VALUE.HP1_ID_BW PARAM_VALUE.GP_ID_BW } {
	# Procedure called to update HP1_ID_BW when any of the dependent parameters in the arguments change
	
	set HP1_ID_BW ${PARAM_VALUE.HP1_ID_BW}
	set GP_ID_BW ${PARAM_VALUE.GP_ID_BW}
	set values(GP_ID_BW) [get_property value $GP_ID_BW]
	set_property value [gen_USERPARAMETER_HP1_ID_BW_VALUE $values(GP_ID_BW)] $HP1_ID_BW
}

proc validate_PARAM_VALUE.HP1_ID_BW { PARAM_VALUE.HP1_ID_BW } {
	# Procedure called to validate HP1_ID_BW
	return true
}

proc update_PARAM_VALUE.HP2_ID_BW { PARAM_VALUE.HP2_ID_BW PARAM_VALUE.GP_ID_BW } {
	# Procedure called to update HP2_ID_BW when any of the dependent parameters in the arguments change
	
	set HP2_ID_BW ${PARAM_VALUE.HP2_ID_BW}
	set GP_ID_BW ${PARAM_VALUE.GP_ID_BW}
	set values(GP_ID_BW) [get_property value $GP_ID_BW]
	set_property value [gen_USERPARAMETER_HP2_ID_BW_VALUE $values(GP_ID_BW)] $HP2_ID_BW
}

proc validate_PARAM_VALUE.HP2_ID_BW { PARAM_VALUE.HP2_ID_BW } {
	# Procedure called to validate HP2_ID_BW
	return true
}

proc update_PARAM_VALUE.HP3_ID_BW { PARAM_VALUE.HP3_ID_BW PARAM_VALUE.GP_ID_BW } {
	# Procedure called to update HP3_ID_BW when any of the dependent parameters in the arguments change
	
	set HP3_ID_BW ${PARAM_VALUE.HP3_ID_BW}
	set GP_ID_BW ${PARAM_VALUE.GP_ID_BW}
	set values(GP_ID_BW) [get_property value $GP_ID_BW]
	set_property value [gen_USERPARAMETER_HP3_ID_BW_VALUE $values(GP_ID_BW)] $HP3_ID_BW
}

proc validate_PARAM_VALUE.HP3_ID_BW { PARAM_VALUE.HP3_ID_BW } {
	# Procedure called to validate HP3_ID_BW
	return true
}

proc update_PARAM_VALUE.HP_DATA_BW { PARAM_VALUE.HP_DATA_BW PARAM_VALUE.ARCH_HP_BW } {
	# Procedure called to update HP_DATA_BW when any of the dependent parameters in the arguments change
	
	set HP_DATA_BW ${PARAM_VALUE.HP_DATA_BW}
	set ARCH_HP_BW ${PARAM_VALUE.ARCH_HP_BW}
	set values(ARCH_HP_BW) [get_property value $ARCH_HP_BW]
	set_property value [gen_USERPARAMETER_HP_DATA_BW_VALUE $values(ARCH_HP_BW)] $HP_DATA_BW
}

proc validate_PARAM_VALUE.HP_DATA_BW { PARAM_VALUE.HP_DATA_BW } {
	# Procedure called to validate HP_DATA_BW
	return true
}

proc update_PARAM_VALUE.M_AXI_AWRLEN_BW { PARAM_VALUE.M_AXI_AWRLEN_BW PARAM_VALUE.AXI_PROTOCOL } {
	# Procedure called to update M_AXI_AWRLEN_BW when any of the dependent parameters in the arguments change
	
	set M_AXI_AWRLEN_BW ${PARAM_VALUE.M_AXI_AWRLEN_BW}
	set AXI_PROTOCOL ${PARAM_VALUE.AXI_PROTOCOL}
	set values(AXI_PROTOCOL) [get_property value $AXI_PROTOCOL]
	set_property value [gen_USERPARAMETER_M_AXI_AWRLEN_BW_VALUE $values(AXI_PROTOCOL)] $M_AXI_AWRLEN_BW
}

proc validate_PARAM_VALUE.M_AXI_AWRLEN_BW { PARAM_VALUE.M_AXI_AWRLEN_BW } {
	# Procedure called to validate M_AXI_AWRLEN_BW
	return true
}

proc update_PARAM_VALUE.M_AXI_AWRLOCK_BW { PARAM_VALUE.M_AXI_AWRLOCK_BW PARAM_VALUE.M_AXI_AWRLEN_BW } {
	# Procedure called to update M_AXI_AWRLOCK_BW when any of the dependent parameters in the arguments change
	
	set M_AXI_AWRLOCK_BW ${PARAM_VALUE.M_AXI_AWRLOCK_BW}
	set M_AXI_AWRLEN_BW ${PARAM_VALUE.M_AXI_AWRLEN_BW}
	set values(M_AXI_AWRLEN_BW) [get_property value $M_AXI_AWRLEN_BW]
	set_property value [gen_USERPARAMETER_M_AXI_AWRLOCK_BW_VALUE $values(M_AXI_AWRLEN_BW)] $M_AXI_AWRLOCK_BW
}

proc validate_PARAM_VALUE.M_AXI_AWRLOCK_BW { PARAM_VALUE.M_AXI_AWRLOCK_BW } {
	# Procedure called to validate M_AXI_AWRLOCK_BW
	return true
}

proc update_PARAM_VALUE.SFM_ENA { PARAM_VALUE.SFM_ENA PARAM_VALUE.M_AXI_AWRLEN_BW } {
	# Procedure called to update SFM_ENA when any of the dependent parameters in the arguments change
	
	set SFM_ENA ${PARAM_VALUE.SFM_ENA}
	set M_AXI_AWRLEN_BW ${PARAM_VALUE.M_AXI_AWRLEN_BW}
	set values(M_AXI_AWRLEN_BW) [get_property value $M_AXI_AWRLEN_BW]
	if { [gen_USERPARAMETER_SFM_ENA_ENABLEMENT $values(M_AXI_AWRLEN_BW)] } {
		set_property enabled true $SFM_ENA
	} else {
		set_property enabled false $SFM_ENA
	}
}

proc validate_PARAM_VALUE.SFM_ENA { PARAM_VALUE.SFM_ENA } {
	# Procedure called to validate SFM_ENA
	return true
}

proc update_PARAM_VALUE.SFM_HP0_ID_BW { PARAM_VALUE.SFM_HP0_ID_BW PARAM_VALUE.GP_ID_BW } {
	# Procedure called to update SFM_HP0_ID_BW when any of the dependent parameters in the arguments change
	
	set SFM_HP0_ID_BW ${PARAM_VALUE.SFM_HP0_ID_BW}
	set GP_ID_BW ${PARAM_VALUE.GP_ID_BW}
	set values(GP_ID_BW) [get_property value $GP_ID_BW]
	set_property value [gen_USERPARAMETER_SFM_HP0_ID_BW_VALUE $values(GP_ID_BW)] $SFM_HP0_ID_BW
}

proc validate_PARAM_VALUE.SFM_HP0_ID_BW { PARAM_VALUE.SFM_HP0_ID_BW } {
	# Procedure called to validate SFM_HP0_ID_BW
	return true
}

proc update_PARAM_VALUE.SUM_AXI_PROTOCOL { PARAM_VALUE.SUM_AXI_PROTOCOL PARAM_VALUE.AXI_PROTOCOL } {
	# Procedure called to update SUM_AXI_PROTOCOL when any of the dependent parameters in the arguments change
	
	set SUM_AXI_PROTOCOL ${PARAM_VALUE.SUM_AXI_PROTOCOL}
	set AXI_PROTOCOL ${PARAM_VALUE.AXI_PROTOCOL}
	set values(AXI_PROTOCOL) [get_property value $AXI_PROTOCOL]
	set_property value [gen_USERPARAMETER_SUM_AXI_PROTOCOL_VALUE $values(AXI_PROTOCOL)] $SUM_AXI_PROTOCOL
}

proc validate_PARAM_VALUE.SUM_AXI_PROTOCOL { PARAM_VALUE.SUM_AXI_PROTOCOL } {
	# Procedure called to validate SUM_AXI_PROTOCOL
	return true
}

proc update_PARAM_VALUE.SUM_BRAM_N { PARAM_VALUE.SUM_BRAM_N PARAM_VALUE.VER_DPU_NUM PARAM_VALUE.BBANK_IMG_N PARAM_VALUE.BBANK_WGT_N PARAM_VALUE.BBANK_BIAS PARAM_VALUE.ARCH_ICP PARAM_VALUE.BANK_IMG_N PARAM_VALUE.SFM_ENA } {
	# Procedure called to update SUM_BRAM_N when any of the dependent parameters in the arguments change
	
	set SUM_BRAM_N ${PARAM_VALUE.SUM_BRAM_N}
	set VER_DPU_NUM ${PARAM_VALUE.VER_DPU_NUM}
	set BBANK_IMG_N ${PARAM_VALUE.BBANK_IMG_N}
	set BBANK_WGT_N ${PARAM_VALUE.BBANK_WGT_N}
	set BBANK_BIAS ${PARAM_VALUE.BBANK_BIAS}
	set ARCH_ICP ${PARAM_VALUE.ARCH_ICP}
	set BANK_IMG_N ${PARAM_VALUE.BANK_IMG_N}
	set SFM_ENA ${PARAM_VALUE.SFM_ENA}
	set values(VER_DPU_NUM) [get_property value $VER_DPU_NUM]
	set values(BBANK_IMG_N) [get_property value $BBANK_IMG_N]
	set values(BBANK_WGT_N) [get_property value $BBANK_WGT_N]
	set values(BBANK_BIAS) [get_property value $BBANK_BIAS]
	set values(ARCH_ICP) [get_property value $ARCH_ICP]
	set values(BANK_IMG_N) [get_property value $BANK_IMG_N]
	set values(SFM_ENA) [get_property value $SFM_ENA]
	set_property value [gen_USERPARAMETER_SUM_BRAM_N_VALUE $values(VER_DPU_NUM) $values(BBANK_IMG_N) $values(BBANK_WGT_N) $values(BBANK_BIAS) $values(ARCH_ICP) $values(BANK_IMG_N) $values(SFM_ENA)] $SUM_BRAM_N
}

proc validate_PARAM_VALUE.SUM_BRAM_N { PARAM_VALUE.SUM_BRAM_N } {
	# Procedure called to validate SUM_BRAM_N
	return true
}

proc update_PARAM_VALUE.SUM_DSP_NUM { PARAM_VALUE.SUM_DSP_NUM PARAM_VALUE.LOAD_DSP_NUM PARAM_VALUE.SAVE_DSP_NUM PARAM_VALUE.CONV_DSP_NUM PARAM_VALUE.DWCV_DSP_NUM PARAM_VALUE.VER_DPU_NUM PARAM_VALUE.SFM_ENA PARAM_VALUE.SFM_DSP_NUM } {
	# Procedure called to update SUM_DSP_NUM when any of the dependent parameters in the arguments change
	
	set SUM_DSP_NUM ${PARAM_VALUE.SUM_DSP_NUM}
	set LOAD_DSP_NUM ${PARAM_VALUE.LOAD_DSP_NUM}
	set SAVE_DSP_NUM ${PARAM_VALUE.SAVE_DSP_NUM}
	set CONV_DSP_NUM ${PARAM_VALUE.CONV_DSP_NUM}
	set DWCV_DSP_NUM ${PARAM_VALUE.DWCV_DSP_NUM}
	set VER_DPU_NUM ${PARAM_VALUE.VER_DPU_NUM}
	set SFM_ENA ${PARAM_VALUE.SFM_ENA}
	set SFM_DSP_NUM ${PARAM_VALUE.SFM_DSP_NUM}
	set values(LOAD_DSP_NUM) [get_property value $LOAD_DSP_NUM]
	set values(SAVE_DSP_NUM) [get_property value $SAVE_DSP_NUM]
	set values(CONV_DSP_NUM) [get_property value $CONV_DSP_NUM]
	set values(DWCV_DSP_NUM) [get_property value $DWCV_DSP_NUM]
	set values(VER_DPU_NUM) [get_property value $VER_DPU_NUM]
	set values(SFM_ENA) [get_property value $SFM_ENA]
	set values(SFM_DSP_NUM) [get_property value $SFM_DSP_NUM]
	set_property value [gen_USERPARAMETER_SUM_DSP_NUM_VALUE $values(LOAD_DSP_NUM) $values(SAVE_DSP_NUM) $values(CONV_DSP_NUM) $values(DWCV_DSP_NUM) $values(VER_DPU_NUM) $values(SFM_ENA) $values(SFM_DSP_NUM)] $SUM_DSP_NUM
}

proc validate_PARAM_VALUE.SUM_DSP_NUM { PARAM_VALUE.SUM_DSP_NUM } {
	# Procedure called to validate SUM_DSP_NUM
	return true
}

proc update_PARAM_VALUE.SUM_HP_DATA_BW { PARAM_VALUE.SUM_HP_DATA_BW PARAM_VALUE.HP_DATA_BW } {
	# Procedure called to update SUM_HP_DATA_BW when any of the dependent parameters in the arguments change
	
	set SUM_HP_DATA_BW ${PARAM_VALUE.SUM_HP_DATA_BW}
	set HP_DATA_BW ${PARAM_VALUE.HP_DATA_BW}
	set values(HP_DATA_BW) [get_property value $HP_DATA_BW]
	set_property value [gen_USERPARAMETER_SUM_HP_DATA_BW_VALUE $values(HP_DATA_BW)] $SUM_HP_DATA_BW
}

proc validate_PARAM_VALUE.SUM_HP_DATA_BW { PARAM_VALUE.SUM_HP_DATA_BW } {
	# Procedure called to validate SUM_HP_DATA_BW
	return true
}

proc update_PARAM_VALUE.SUM_SFM_HP_DATA_BW { PARAM_VALUE.SUM_SFM_HP_DATA_BW PARAM_VALUE.SFM_HP_DATA_BW } {
	# Procedure called to update SUM_SFM_HP_DATA_BW when any of the dependent parameters in the arguments change
	
	set SUM_SFM_HP_DATA_BW ${PARAM_VALUE.SUM_SFM_HP_DATA_BW}
	set SFM_HP_DATA_BW ${PARAM_VALUE.SFM_HP_DATA_BW}
	set values(SFM_HP_DATA_BW) [get_property value $SFM_HP_DATA_BW]
	set_property value [gen_USERPARAMETER_SUM_SFM_HP_DATA_BW_VALUE $values(SFM_HP_DATA_BW)] $SUM_SFM_HP_DATA_BW
}

proc validate_PARAM_VALUE.SUM_SFM_HP_DATA_BW { PARAM_VALUE.SUM_SFM_HP_DATA_BW } {
	# Procedure called to validate SUM_SFM_HP_DATA_BW
	return true
}

proc update_PARAM_VALUE.SUM_URAM_N { PARAM_VALUE.SUM_URAM_N PARAM_VALUE.VER_DPU_NUM PARAM_VALUE.UBANK_IMG_N PARAM_VALUE.UBANK_WGT_N PARAM_VALUE.UBANK_BIAS PARAM_VALUE.ARCH_ICP } {
	# Procedure called to update SUM_URAM_N when any of the dependent parameters in the arguments change
	
	set SUM_URAM_N ${PARAM_VALUE.SUM_URAM_N}
	set VER_DPU_NUM ${PARAM_VALUE.VER_DPU_NUM}
	set UBANK_IMG_N ${PARAM_VALUE.UBANK_IMG_N}
	set UBANK_WGT_N ${PARAM_VALUE.UBANK_WGT_N}
	set UBANK_BIAS ${PARAM_VALUE.UBANK_BIAS}
	set ARCH_ICP ${PARAM_VALUE.ARCH_ICP}
	set values(VER_DPU_NUM) [get_property value $VER_DPU_NUM]
	set values(UBANK_IMG_N) [get_property value $UBANK_IMG_N]
	set values(UBANK_WGT_N) [get_property value $UBANK_WGT_N]
	set values(UBANK_BIAS) [get_property value $UBANK_BIAS]
	set values(ARCH_ICP) [get_property value $ARCH_ICP]
	set_property value [gen_USERPARAMETER_SUM_URAM_N_VALUE $values(VER_DPU_NUM) $values(UBANK_IMG_N) $values(UBANK_WGT_N) $values(UBANK_BIAS) $values(ARCH_ICP)] $SUM_URAM_N
}

proc validate_PARAM_VALUE.SUM_URAM_N { PARAM_VALUE.SUM_URAM_N } {
	# Procedure called to validate SUM_URAM_N
	return true
}

proc update_PARAM_VALUE.S_AXI_AWRLEN_BW { PARAM_VALUE.S_AXI_AWRLEN_BW PARAM_VALUE.AXI_PROTOCOL } {
	# Procedure called to update S_AXI_AWRLEN_BW when any of the dependent parameters in the arguments change
	
	set S_AXI_AWRLEN_BW ${PARAM_VALUE.S_AXI_AWRLEN_BW}
	set AXI_PROTOCOL ${PARAM_VALUE.AXI_PROTOCOL}
	set values(AXI_PROTOCOL) [get_property value $AXI_PROTOCOL]
	set_property value [gen_USERPARAMETER_S_AXI_AWRLEN_BW_VALUE $values(AXI_PROTOCOL)] $S_AXI_AWRLEN_BW
}

proc validate_PARAM_VALUE.S_AXI_AWRLEN_BW { PARAM_VALUE.S_AXI_AWRLEN_BW } {
	# Procedure called to validate S_AXI_AWRLEN_BW
	return true
}

proc update_PARAM_VALUE.S_AXI_ID_BW { PARAM_VALUE.S_AXI_ID_BW PARAM_VALUE.M_AXI_AWRLEN_BW } {
	# Procedure called to update S_AXI_ID_BW when any of the dependent parameters in the arguments change
	
	set S_AXI_ID_BW ${PARAM_VALUE.S_AXI_ID_BW}
	set M_AXI_AWRLEN_BW ${PARAM_VALUE.M_AXI_AWRLEN_BW}
	set values(M_AXI_AWRLEN_BW) [get_property value $M_AXI_AWRLEN_BW]
	set_property value [gen_USERPARAMETER_S_AXI_ID_BW_VALUE $values(M_AXI_AWRLEN_BW)] $S_AXI_ID_BW
}

proc validate_PARAM_VALUE.S_AXI_ID_BW { PARAM_VALUE.S_AXI_ID_BW } {
	# Procedure called to validate S_AXI_ID_BW
	return true
}

proc update_PARAM_VALUE.UBANK_BIAS { PARAM_VALUE.UBANK_BIAS PARAM_VALUE.UBANK_BIAS_USER } {
	# Procedure called to update UBANK_BIAS when any of the dependent parameters in the arguments change
	
	set UBANK_BIAS ${PARAM_VALUE.UBANK_BIAS}
	set UBANK_BIAS_USER ${PARAM_VALUE.UBANK_BIAS_USER}
	set values(UBANK_BIAS_USER) [get_property value $UBANK_BIAS_USER]
	set_property value [gen_USERPARAMETER_UBANK_BIAS_VALUE $values(UBANK_BIAS_USER)] $UBANK_BIAS
}

proc validate_PARAM_VALUE.UBANK_BIAS { PARAM_VALUE.UBANK_BIAS } {
	# Procedure called to validate UBANK_BIAS
	return true
}

proc update_PARAM_VALUE.UBANK_BIAS_USER { PARAM_VALUE.UBANK_BIAS_USER PARAM_VALUE.URAM_N_USER PARAM_VALUE.ARCH_ICP } {
	# Procedure called to update UBANK_BIAS_USER when any of the dependent parameters in the arguments change
	
	set UBANK_BIAS_USER ${PARAM_VALUE.UBANK_BIAS_USER}
	set URAM_N_USER ${PARAM_VALUE.URAM_N_USER}
	set ARCH_ICP ${PARAM_VALUE.ARCH_ICP}
	set values(URAM_N_USER) [get_property value $URAM_N_USER]
	set values(ARCH_ICP) [get_property value $ARCH_ICP]
	set_property value [gen_USERPARAMETER_UBANK_BIAS_USER_VALUE $values(URAM_N_USER) $values(ARCH_ICP)] $UBANK_BIAS_USER
}

proc validate_PARAM_VALUE.UBANK_BIAS_USER { PARAM_VALUE.UBANK_BIAS_USER } {
	# Procedure called to validate UBANK_BIAS_USER
	return true
}

proc update_PARAM_VALUE.UBANK_IMG_N { PARAM_VALUE.UBANK_IMG_N PARAM_VALUE.UBANK_IMG_N_USER } {
	# Procedure called to update UBANK_IMG_N when any of the dependent parameters in the arguments change
	
	set UBANK_IMG_N ${PARAM_VALUE.UBANK_IMG_N}
	set UBANK_IMG_N_USER ${PARAM_VALUE.UBANK_IMG_N_USER}
	set values(UBANK_IMG_N_USER) [get_property value $UBANK_IMG_N_USER]
	set_property value [gen_USERPARAMETER_UBANK_IMG_N_VALUE $values(UBANK_IMG_N_USER)] $UBANK_IMG_N
}

proc validate_PARAM_VALUE.UBANK_IMG_N { PARAM_VALUE.UBANK_IMG_N } {
	# Procedure called to validate UBANK_IMG_N
	return true
}

proc update_PARAM_VALUE.UBANK_IMG_N_USER { PARAM_VALUE.UBANK_IMG_N_USER PARAM_VALUE.URAM_N_USER PARAM_VALUE.ARCH_OCP PARAM_VALUE.ARCH_ICP PARAM_VALUE.ARCH_PP PARAM_VALUE.ARCH_IMG_BKGRP } {
	# Procedure called to update UBANK_IMG_N_USER when any of the dependent parameters in the arguments change
	
	set UBANK_IMG_N_USER ${PARAM_VALUE.UBANK_IMG_N_USER}
	set URAM_N_USER ${PARAM_VALUE.URAM_N_USER}
	set ARCH_OCP ${PARAM_VALUE.ARCH_OCP}
	set ARCH_ICP ${PARAM_VALUE.ARCH_ICP}
	set ARCH_PP ${PARAM_VALUE.ARCH_PP}
	set ARCH_IMG_BKGRP ${PARAM_VALUE.ARCH_IMG_BKGRP}
	set values(URAM_N_USER) [get_property value $URAM_N_USER]
	set values(ARCH_OCP) [get_property value $ARCH_OCP]
	set values(ARCH_ICP) [get_property value $ARCH_ICP]
	set values(ARCH_PP) [get_property value $ARCH_PP]
	set values(ARCH_IMG_BKGRP) [get_property value $ARCH_IMG_BKGRP]
	set_property value [gen_USERPARAMETER_UBANK_IMG_N_USER_VALUE $values(URAM_N_USER) $values(ARCH_OCP) $values(ARCH_ICP) $values(ARCH_PP) $values(ARCH_IMG_BKGRP)] $UBANK_IMG_N_USER
}

proc validate_PARAM_VALUE.UBANK_IMG_N_USER { PARAM_VALUE.UBANK_IMG_N_USER } {
	# Procedure called to validate UBANK_IMG_N_USER
	return true
}

proc update_PARAM_VALUE.UBANK_WGT_N { PARAM_VALUE.UBANK_WGT_N PARAM_VALUE.UBANK_WGT_N_USER } {
	# Procedure called to update UBANK_WGT_N when any of the dependent parameters in the arguments change
	
	set UBANK_WGT_N ${PARAM_VALUE.UBANK_WGT_N}
	set UBANK_WGT_N_USER ${PARAM_VALUE.UBANK_WGT_N_USER}
	set values(UBANK_WGT_N_USER) [get_property value $UBANK_WGT_N_USER]
	set_property value [gen_USERPARAMETER_UBANK_WGT_N_VALUE $values(UBANK_WGT_N_USER)] $UBANK_WGT_N
}

proc validate_PARAM_VALUE.UBANK_WGT_N { PARAM_VALUE.UBANK_WGT_N } {
	# Procedure called to validate UBANK_WGT_N
	return true
}

proc update_PARAM_VALUE.UBANK_WGT_N_USER { PARAM_VALUE.UBANK_WGT_N_USER PARAM_VALUE.URAM_N_USER PARAM_VALUE.ARCH_ICP PARAM_VALUE.ARCH_OCP PARAM_VALUE.DWCV_ENA } {
	# Procedure called to update UBANK_WGT_N_USER when any of the dependent parameters in the arguments change
	
	set UBANK_WGT_N_USER ${PARAM_VALUE.UBANK_WGT_N_USER}
	set URAM_N_USER ${PARAM_VALUE.URAM_N_USER}
	set ARCH_ICP ${PARAM_VALUE.ARCH_ICP}
	set ARCH_OCP ${PARAM_VALUE.ARCH_OCP}
	set DWCV_ENA ${PARAM_VALUE.DWCV_ENA}
	set values(URAM_N_USER) [get_property value $URAM_N_USER]
	set values(ARCH_ICP) [get_property value $ARCH_ICP]
	set values(ARCH_OCP) [get_property value $ARCH_OCP]
	set values(DWCV_ENA) [get_property value $DWCV_ENA]
	set_property value [gen_USERPARAMETER_UBANK_WGT_N_USER_VALUE $values(URAM_N_USER) $values(ARCH_ICP) $values(ARCH_OCP) $values(DWCV_ENA)] $UBANK_WGT_N_USER
}

proc validate_PARAM_VALUE.UBANK_WGT_N_USER { PARAM_VALUE.UBANK_WGT_N_USER } {
	# Procedure called to validate UBANK_WGT_N_USER
	return true
}

proc update_PARAM_VALUE.ARCH { PARAM_VALUE.ARCH } {
	# Procedure called to update ARCH when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.ARCH { PARAM_VALUE.ARCH } {
	# Procedure called to validate ARCH
	return true
}

proc update_PARAM_VALUE.ARCH_DATA_BW { PARAM_VALUE.ARCH_DATA_BW } {
	# Procedure called to update ARCH_DATA_BW when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.ARCH_DATA_BW { PARAM_VALUE.ARCH_DATA_BW } {
	# Procedure called to validate ARCH_DATA_BW
	return true
}

proc update_PARAM_VALUE.ARCH_HP_BW { PARAM_VALUE.ARCH_HP_BW } {
	# Procedure called to update ARCH_HP_BW when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.ARCH_HP_BW { PARAM_VALUE.ARCH_HP_BW } {
	# Procedure called to validate ARCH_HP_BW
	return true
}

proc update_PARAM_VALUE.ARCH_IMG_BKGRP { PARAM_VALUE.ARCH_IMG_BKGRP } {
	# Procedure called to update ARCH_IMG_BKGRP when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.ARCH_IMG_BKGRP { PARAM_VALUE.ARCH_IMG_BKGRP } {
	# Procedure called to validate ARCH_IMG_BKGRP
	return true
}

proc update_PARAM_VALUE.AXI_PROTOCOL { PARAM_VALUE.AXI_PROTOCOL } {
	# Procedure called to update AXI_PROTOCOL when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.AXI_PROTOCOL { PARAM_VALUE.AXI_PROTOCOL } {
	# Procedure called to validate AXI_PROTOCOL
	return true
}

proc update_PARAM_VALUE.BANK_BIAS { PARAM_VALUE.BANK_BIAS } {
	# Procedure called to update BANK_BIAS when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.BANK_BIAS { PARAM_VALUE.BANK_BIAS } {
	# Procedure called to validate BANK_BIAS
	return true
}

proc update_PARAM_VALUE.CONV_DSP_ACCU_ENA { PARAM_VALUE.CONV_DSP_ACCU_ENA } {
	# Procedure called to update CONV_DSP_ACCU_ENA when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.CONV_DSP_ACCU_ENA { PARAM_VALUE.CONV_DSP_ACCU_ENA } {
	# Procedure called to validate CONV_DSP_ACCU_ENA
	return true
}

proc update_PARAM_VALUE.CONV_RELU_ADDON { PARAM_VALUE.CONV_RELU_ADDON } {
	# Procedure called to update CONV_RELU_ADDON when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.CONV_RELU_ADDON { PARAM_VALUE.CONV_RELU_ADDON } {
	# Procedure called to validate CONV_RELU_ADDON
	return true
}

proc update_PARAM_VALUE.CONV_WR_PARALLEL { PARAM_VALUE.CONV_WR_PARALLEL } {
	# Procedure called to update CONV_WR_PARALLEL when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.CONV_WR_PARALLEL { PARAM_VALUE.CONV_WR_PARALLEL } {
	# Procedure called to validate CONV_WR_PARALLEL
	return true
}

proc update_PARAM_VALUE.DBANK_BIAS { PARAM_VALUE.DBANK_BIAS } {
	# Procedure called to update DBANK_BIAS when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.DBANK_BIAS { PARAM_VALUE.DBANK_BIAS } {
	# Procedure called to validate DBANK_BIAS
	return true
}

proc update_PARAM_VALUE.DBANK_IMG_N { PARAM_VALUE.DBANK_IMG_N } {
	# Procedure called to update DBANK_IMG_N when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.DBANK_IMG_N { PARAM_VALUE.DBANK_IMG_N } {
	# Procedure called to validate DBANK_IMG_N
	return true
}

proc update_PARAM_VALUE.DBANK_WGT_N { PARAM_VALUE.DBANK_WGT_N } {
	# Procedure called to update DBANK_WGT_N when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.DBANK_WGT_N { PARAM_VALUE.DBANK_WGT_N } {
	# Procedure called to validate DBANK_WGT_N
	return true
}

proc update_PARAM_VALUE.DSP48_VER { PARAM_VALUE.DSP48_VER } {
	# Procedure called to update DSP48_VER when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.DSP48_VER { PARAM_VALUE.DSP48_VER } {
	# Procedure called to validate DSP48_VER
	return true
}

proc update_PARAM_VALUE.DWCV_ALU_MODE { PARAM_VALUE.DWCV_ALU_MODE } {
	# Procedure called to update DWCV_ALU_MODE when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.DWCV_ALU_MODE { PARAM_VALUE.DWCV_ALU_MODE } {
	# Procedure called to validate DWCV_ALU_MODE
	return true
}

proc update_PARAM_VALUE.DWCV_ENA { PARAM_VALUE.DWCV_ENA } {
	# Procedure called to update DWCV_ENA when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.DWCV_ENA { PARAM_VALUE.DWCV_ENA } {
	# Procedure called to validate DWCV_ENA
	return true
}

proc update_PARAM_VALUE.ELEW_MULT_EN { PARAM_VALUE.ELEW_MULT_EN } {
	# Procedure called to update ELEW_MULT_EN when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.ELEW_MULT_EN { PARAM_VALUE.ELEW_MULT_EN } {
	# Procedure called to validate ELEW_MULT_EN
	return true
}

proc update_PARAM_VALUE.ELEW_PARALLEL { PARAM_VALUE.ELEW_PARALLEL } {
	# Procedure called to update ELEW_PARALLEL when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.ELEW_PARALLEL { PARAM_VALUE.ELEW_PARALLEL } {
	# Procedure called to validate ELEW_PARALLEL
	return true
}

proc update_PARAM_VALUE.GIT_COMMIT_ID { PARAM_VALUE.GIT_COMMIT_ID } {
	# Procedure called to update GIT_COMMIT_ID when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.GIT_COMMIT_ID { PARAM_VALUE.GIT_COMMIT_ID } {
	# Procedure called to validate GIT_COMMIT_ID
	return true
}

proc update_PARAM_VALUE.GIT_COMMIT_TIME { PARAM_VALUE.GIT_COMMIT_TIME } {
	# Procedure called to update GIT_COMMIT_TIME when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.GIT_COMMIT_TIME { PARAM_VALUE.GIT_COMMIT_TIME } {
	# Procedure called to validate GIT_COMMIT_TIME
	return true
}

proc update_PARAM_VALUE.GP_ID_BW { PARAM_VALUE.GP_ID_BW } {
	# Procedure called to update GP_ID_BW when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.GP_ID_BW { PARAM_VALUE.GP_ID_BW } {
	# Procedure called to validate GP_ID_BW
	return true
}

proc update_PARAM_VALUE.LOAD_AUGM { PARAM_VALUE.LOAD_AUGM } {
	# Procedure called to update LOAD_AUGM when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.LOAD_AUGM { PARAM_VALUE.LOAD_AUGM } {
	# Procedure called to validate LOAD_AUGM
	return true
}

proc update_PARAM_VALUE.LOAD_DSP_NUM { PARAM_VALUE.LOAD_DSP_NUM } {
	# Procedure called to update LOAD_DSP_NUM when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.LOAD_DSP_NUM { PARAM_VALUE.LOAD_DSP_NUM } {
	# Procedure called to validate LOAD_DSP_NUM
	return true
}

proc update_PARAM_VALUE.LOAD_IMG_MEAN { PARAM_VALUE.LOAD_IMG_MEAN } {
	# Procedure called to update LOAD_IMG_MEAN when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.LOAD_IMG_MEAN { PARAM_VALUE.LOAD_IMG_MEAN } {
	# Procedure called to validate LOAD_IMG_MEAN
	return true
}

proc update_PARAM_VALUE.LOAD_PARALLEL { PARAM_VALUE.LOAD_PARALLEL } {
	# Procedure called to update LOAD_PARALLEL when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.LOAD_PARALLEL { PARAM_VALUE.LOAD_PARALLEL } {
	# Procedure called to validate LOAD_PARALLEL
	return true
}

proc update_PARAM_VALUE.MISC_WR_PARALLEL { PARAM_VALUE.MISC_WR_PARALLEL } {
	# Procedure called to update MISC_WR_PARALLEL when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.MISC_WR_PARALLEL { PARAM_VALUE.MISC_WR_PARALLEL } {
	# Procedure called to validate MISC_WR_PARALLEL
	return true
}

proc update_PARAM_VALUE.M_AXI_AWRUSER_BW { PARAM_VALUE.M_AXI_AWRUSER_BW } {
	# Procedure called to update M_AXI_AWRUSER_BW when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.M_AXI_AWRUSER_BW { PARAM_VALUE.M_AXI_AWRUSER_BW } {
	# Procedure called to validate M_AXI_AWRUSER_BW
	return true
}

proc update_PARAM_VALUE.M_AXI_FREQ_MHZ { PARAM_VALUE.M_AXI_FREQ_MHZ } {
	# Procedure called to update M_AXI_FREQ_MHZ when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.M_AXI_FREQ_MHZ { PARAM_VALUE.M_AXI_FREQ_MHZ } {
	# Procedure called to validate M_AXI_FREQ_MHZ
	return true
}

proc update_PARAM_VALUE.POOL_AVERAGE { PARAM_VALUE.POOL_AVERAGE } {
	# Procedure called to update POOL_AVERAGE when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.POOL_AVERAGE { PARAM_VALUE.POOL_AVERAGE } {
	# Procedure called to validate POOL_AVERAGE
	return true
}

proc update_PARAM_VALUE.RAM_DEPTH_BIAS { PARAM_VALUE.RAM_DEPTH_BIAS } {
	# Procedure called to update RAM_DEPTH_BIAS when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.RAM_DEPTH_BIAS { PARAM_VALUE.RAM_DEPTH_BIAS } {
	# Procedure called to validate RAM_DEPTH_BIAS
	return true
}

proc update_PARAM_VALUE.RAM_DEPTH_IMG { PARAM_VALUE.RAM_DEPTH_IMG } {
	# Procedure called to update RAM_DEPTH_IMG when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.RAM_DEPTH_IMG { PARAM_VALUE.RAM_DEPTH_IMG } {
	# Procedure called to validate RAM_DEPTH_IMG
	return true
}

proc update_PARAM_VALUE.RAM_DEPTH_MEAN { PARAM_VALUE.RAM_DEPTH_MEAN } {
	# Procedure called to update RAM_DEPTH_MEAN when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.RAM_DEPTH_MEAN { PARAM_VALUE.RAM_DEPTH_MEAN } {
	# Procedure called to validate RAM_DEPTH_MEAN
	return true
}

proc update_PARAM_VALUE.RAM_DEPTH_WGT { PARAM_VALUE.RAM_DEPTH_WGT } {
	# Procedure called to update RAM_DEPTH_WGT when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.RAM_DEPTH_WGT { PARAM_VALUE.RAM_DEPTH_WGT } {
	# Procedure called to validate RAM_DEPTH_WGT
	return true
}

proc update_PARAM_VALUE.SAVE_DSP_NUM { PARAM_VALUE.SAVE_DSP_NUM } {
	# Procedure called to update SAVE_DSP_NUM when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.SAVE_DSP_NUM { PARAM_VALUE.SAVE_DSP_NUM } {
	# Procedure called to validate SAVE_DSP_NUM
	return true
}

proc update_PARAM_VALUE.SAVE_PARALLEL { PARAM_VALUE.SAVE_PARALLEL } {
	# Procedure called to update SAVE_PARALLEL when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.SAVE_PARALLEL { PARAM_VALUE.SAVE_PARALLEL } {
	# Procedure called to validate SAVE_PARALLEL
	return true
}

proc update_PARAM_VALUE.SFM_DSP_NUM { PARAM_VALUE.SFM_DSP_NUM } {
	# Procedure called to update SFM_DSP_NUM when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.SFM_DSP_NUM { PARAM_VALUE.SFM_DSP_NUM } {
	# Procedure called to validate SFM_DSP_NUM
	return true
}

proc update_PARAM_VALUE.SFM_HP_DATA_BW { PARAM_VALUE.SFM_HP_DATA_BW } {
	# Procedure called to update SFM_HP_DATA_BW when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.SFM_HP_DATA_BW { PARAM_VALUE.SFM_HP_DATA_BW } {
	# Procedure called to validate SFM_HP_DATA_BW
	return true
}

proc update_PARAM_VALUE.SUM_GP_DATA_BW { PARAM_VALUE.SUM_GP_DATA_BW } {
	# Procedure called to update SUM_GP_DATA_BW when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.SUM_GP_DATA_BW { PARAM_VALUE.SUM_GP_DATA_BW } {
	# Procedure called to validate SUM_GP_DATA_BW
	return true
}

proc update_PARAM_VALUE.SUM_S_AXI_DATA_BW { PARAM_VALUE.SUM_S_AXI_DATA_BW } {
	# Procedure called to update SUM_S_AXI_DATA_BW when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.SUM_S_AXI_DATA_BW { PARAM_VALUE.SUM_S_AXI_DATA_BW } {
	# Procedure called to validate SUM_S_AXI_DATA_BW
	return true
}

proc update_PARAM_VALUE.SUM_VER_TARGET { PARAM_VALUE.SUM_VER_TARGET } {
	# Procedure called to update SUM_VER_TARGET when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.SUM_VER_TARGET { PARAM_VALUE.SUM_VER_TARGET } {
	# Procedure called to validate SUM_VER_TARGET
	return true
}

proc update_PARAM_VALUE.SYS_IP_TYPE { PARAM_VALUE.SYS_IP_TYPE } {
	# Procedure called to update SYS_IP_TYPE when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.SYS_IP_TYPE { PARAM_VALUE.SYS_IP_TYPE } {
	# Procedure called to validate SYS_IP_TYPE
	return true
}

proc update_PARAM_VALUE.SYS_IP_VER { PARAM_VALUE.SYS_IP_VER } {
	# Procedure called to update SYS_IP_VER when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.SYS_IP_VER { PARAM_VALUE.SYS_IP_VER } {
	# Procedure called to validate SYS_IP_VER
	return true
}

proc update_PARAM_VALUE.SYS_REGMAP_SIZE { PARAM_VALUE.SYS_REGMAP_SIZE } {
	# Procedure called to update SYS_REGMAP_SIZE when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.SYS_REGMAP_SIZE { PARAM_VALUE.SYS_REGMAP_SIZE } {
	# Procedure called to validate SYS_REGMAP_SIZE
	return true
}

proc update_PARAM_VALUE.SYS_REGMAP_VER { PARAM_VALUE.SYS_REGMAP_VER } {
	# Procedure called to update SYS_REGMAP_VER when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.SYS_REGMAP_VER { PARAM_VALUE.SYS_REGMAP_VER } {
	# Procedure called to validate SYS_REGMAP_VER
	return true
}

proc update_PARAM_VALUE.S_AXI_CLK_INDEPENDENT { PARAM_VALUE.S_AXI_CLK_INDEPENDENT } {
	# Procedure called to update S_AXI_CLK_INDEPENDENT when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.S_AXI_CLK_INDEPENDENT { PARAM_VALUE.S_AXI_CLK_INDEPENDENT } {
	# Procedure called to validate S_AXI_CLK_INDEPENDENT
	return true
}

proc update_PARAM_VALUE.S_AXI_FREQ_MHZ { PARAM_VALUE.S_AXI_FREQ_MHZ } {
	# Procedure called to update S_AXI_FREQ_MHZ when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.S_AXI_FREQ_MHZ { PARAM_VALUE.S_AXI_FREQ_MHZ } {
	# Procedure called to validate S_AXI_FREQ_MHZ
	return true
}

proc update_PARAM_VALUE.S_AXI_SLAVES_BASE_ADDR { PARAM_VALUE.S_AXI_SLAVES_BASE_ADDR } {
	# Procedure called to update S_AXI_SLAVES_BASE_ADDR when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.S_AXI_SLAVES_BASE_ADDR { PARAM_VALUE.S_AXI_SLAVES_BASE_ADDR } {
	# Procedure called to validate S_AXI_SLAVES_BASE_ADDR
	return true
}

proc update_PARAM_VALUE.TIMESTAMP_ENA { PARAM_VALUE.TIMESTAMP_ENA } {
	# Procedure called to update TIMESTAMP_ENA when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.TIMESTAMP_ENA { PARAM_VALUE.TIMESTAMP_ENA } {
	# Procedure called to validate TIMESTAMP_ENA
	return true
}

proc update_PARAM_VALUE.TIME_DAY { PARAM_VALUE.TIME_DAY } {
	# Procedure called to update TIME_DAY when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.TIME_DAY { PARAM_VALUE.TIME_DAY } {
	# Procedure called to validate TIME_DAY
	return true
}

proc update_PARAM_VALUE.TIME_HOUR { PARAM_VALUE.TIME_HOUR } {
	# Procedure called to update TIME_HOUR when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.TIME_HOUR { PARAM_VALUE.TIME_HOUR } {
	# Procedure called to validate TIME_HOUR
	return true
}

proc update_PARAM_VALUE.TIME_MONTH { PARAM_VALUE.TIME_MONTH } {
	# Procedure called to update TIME_MONTH when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.TIME_MONTH { PARAM_VALUE.TIME_MONTH } {
	# Procedure called to validate TIME_MONTH
	return true
}

proc update_PARAM_VALUE.TIME_QUARTER { PARAM_VALUE.TIME_QUARTER } {
	# Procedure called to update TIME_QUARTER when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.TIME_QUARTER { PARAM_VALUE.TIME_QUARTER } {
	# Procedure called to validate TIME_QUARTER
	return true
}

proc update_PARAM_VALUE.TIME_YEAR { PARAM_VALUE.TIME_YEAR } {
	# Procedure called to update TIME_YEAR when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.TIME_YEAR { PARAM_VALUE.TIME_YEAR } {
	# Procedure called to validate TIME_YEAR
	return true
}

proc update_PARAM_VALUE.URAM_N_USER { PARAM_VALUE.URAM_N_USER } {
	# Procedure called to update URAM_N_USER when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.URAM_N_USER { PARAM_VALUE.URAM_N_USER } {
	# Procedure called to validate URAM_N_USER
	return true
}

proc update_PARAM_VALUE.VER_CHIP_PART { PARAM_VALUE.VER_CHIP_PART } {
	# Procedure called to update VER_CHIP_PART when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.VER_CHIP_PART { PARAM_VALUE.VER_CHIP_PART } {
	# Procedure called to validate VER_CHIP_PART
	return true
}

proc update_PARAM_VALUE.VER_DPU_NUM { PARAM_VALUE.VER_DPU_NUM } {
	# Procedure called to update VER_DPU_NUM when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.VER_DPU_NUM { PARAM_VALUE.VER_DPU_NUM } {
	# Procedure called to validate VER_DPU_NUM
	return true
}

proc update_PARAM_VALUE.VER_IP_REV { PARAM_VALUE.VER_IP_REV } {
	# Procedure called to update VER_IP_REV when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.VER_IP_REV { PARAM_VALUE.VER_IP_REV } {
	# Procedure called to validate VER_IP_REV
	return true
}

proc update_PARAM_VALUE.VER_TARGET { PARAM_VALUE.VER_TARGET } {
	# Procedure called to update VER_TARGET when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.VER_TARGET { PARAM_VALUE.VER_TARGET } {
	# Procedure called to validate VER_TARGET
	return true
}


proc update_MODELPARAM_VALUE.VER_CHIP_PART { MODELPARAM_VALUE.VER_CHIP_PART PARAM_VALUE.VER_CHIP_PART } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.VER_CHIP_PART}] ${MODELPARAM_VALUE.VER_CHIP_PART}
}

proc update_MODELPARAM_VALUE.VER_DPU_NUM { MODELPARAM_VALUE.VER_DPU_NUM PARAM_VALUE.VER_DPU_NUM } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.VER_DPU_NUM}] ${MODELPARAM_VALUE.VER_DPU_NUM}
}

proc update_MODELPARAM_VALUE.CLK_GATING_ENA { MODELPARAM_VALUE.CLK_GATING_ENA PARAM_VALUE.CLK_GATING_ENA } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.CLK_GATING_ENA}] ${MODELPARAM_VALUE.CLK_GATING_ENA}
}

proc update_MODELPARAM_VALUE.DSP48_VER { MODELPARAM_VALUE.DSP48_VER PARAM_VALUE.DSP48_VER } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.DSP48_VER}] ${MODELPARAM_VALUE.DSP48_VER}
}

proc update_MODELPARAM_VALUE.S_AXI_FREQ_MHZ { MODELPARAM_VALUE.S_AXI_FREQ_MHZ PARAM_VALUE.S_AXI_FREQ_MHZ } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.S_AXI_FREQ_MHZ}] ${MODELPARAM_VALUE.S_AXI_FREQ_MHZ}
}

proc update_MODELPARAM_VALUE.S_AXI_CLK_INDEPENDENT { MODELPARAM_VALUE.S_AXI_CLK_INDEPENDENT PARAM_VALUE.S_AXI_CLK_INDEPENDENT } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.S_AXI_CLK_INDEPENDENT}] ${MODELPARAM_VALUE.S_AXI_CLK_INDEPENDENT}
}

proc update_MODELPARAM_VALUE.S_AXI_SLAVES_BASE_ADDR { MODELPARAM_VALUE.S_AXI_SLAVES_BASE_ADDR PARAM_VALUE.S_AXI_SLAVES_BASE_ADDR } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.S_AXI_SLAVES_BASE_ADDR}] ${MODELPARAM_VALUE.S_AXI_SLAVES_BASE_ADDR}
}

proc update_MODELPARAM_VALUE.S_AXI_ID_BW { MODELPARAM_VALUE.S_AXI_ID_BW PARAM_VALUE.S_AXI_ID_BW } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.S_AXI_ID_BW}] ${MODELPARAM_VALUE.S_AXI_ID_BW}
}

proc update_MODELPARAM_VALUE.S_AXI_AWRLEN_BW { MODELPARAM_VALUE.S_AXI_AWRLEN_BW PARAM_VALUE.S_AXI_AWRLEN_BW } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.S_AXI_AWRLEN_BW}] ${MODELPARAM_VALUE.S_AXI_AWRLEN_BW}
}

proc update_MODELPARAM_VALUE.M_AXI_FREQ_MHZ { MODELPARAM_VALUE.M_AXI_FREQ_MHZ PARAM_VALUE.M_AXI_FREQ_MHZ } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.M_AXI_FREQ_MHZ}] ${MODELPARAM_VALUE.M_AXI_FREQ_MHZ}
}

proc update_MODELPARAM_VALUE.M_AXI_AWRLEN_BW { MODELPARAM_VALUE.M_AXI_AWRLEN_BW PARAM_VALUE.M_AXI_AWRLEN_BW } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.M_AXI_AWRLEN_BW}] ${MODELPARAM_VALUE.M_AXI_AWRLEN_BW}
}

proc update_MODELPARAM_VALUE.M_AXI_AWRUSER_BW { MODELPARAM_VALUE.M_AXI_AWRUSER_BW PARAM_VALUE.M_AXI_AWRUSER_BW } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.M_AXI_AWRUSER_BW}] ${MODELPARAM_VALUE.M_AXI_AWRUSER_BW}
}

proc update_MODELPARAM_VALUE.M_AXI_AWRLOCK_BW { MODELPARAM_VALUE.M_AXI_AWRLOCK_BW PARAM_VALUE.M_AXI_AWRLOCK_BW } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.M_AXI_AWRLOCK_BW}] ${MODELPARAM_VALUE.M_AXI_AWRLOCK_BW}
}

proc update_MODELPARAM_VALUE.GP_ID_BW { MODELPARAM_VALUE.GP_ID_BW PARAM_VALUE.GP_ID_BW } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.GP_ID_BW}] ${MODELPARAM_VALUE.GP_ID_BW}
}

proc update_MODELPARAM_VALUE.HP0_ID_BW { MODELPARAM_VALUE.HP0_ID_BW PARAM_VALUE.HP0_ID_BW } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.HP0_ID_BW}] ${MODELPARAM_VALUE.HP0_ID_BW}
}

proc update_MODELPARAM_VALUE.HP1_ID_BW { MODELPARAM_VALUE.HP1_ID_BW PARAM_VALUE.HP1_ID_BW } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.HP1_ID_BW}] ${MODELPARAM_VALUE.HP1_ID_BW}
}

proc update_MODELPARAM_VALUE.HP2_ID_BW { MODELPARAM_VALUE.HP2_ID_BW PARAM_VALUE.HP2_ID_BW } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.HP2_ID_BW}] ${MODELPARAM_VALUE.HP2_ID_BW}
}

proc update_MODELPARAM_VALUE.HP3_ID_BW { MODELPARAM_VALUE.HP3_ID_BW PARAM_VALUE.HP3_ID_BW } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.HP3_ID_BW}] ${MODELPARAM_VALUE.HP3_ID_BW}
}

proc update_MODELPARAM_VALUE.HP_DATA_BW { MODELPARAM_VALUE.HP_DATA_BW PARAM_VALUE.HP_DATA_BW } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.HP_DATA_BW}] ${MODELPARAM_VALUE.HP_DATA_BW}
}

proc update_MODELPARAM_VALUE.SYS_IP_VER { MODELPARAM_VALUE.SYS_IP_VER PARAM_VALUE.SYS_IP_VER } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.SYS_IP_VER}] ${MODELPARAM_VALUE.SYS_IP_VER}
}

proc update_MODELPARAM_VALUE.SYS_IP_TYPE { MODELPARAM_VALUE.SYS_IP_TYPE PARAM_VALUE.SYS_IP_TYPE } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.SYS_IP_TYPE}] ${MODELPARAM_VALUE.SYS_IP_TYPE}
}

proc update_MODELPARAM_VALUE.SYS_REGMAP_SIZE { MODELPARAM_VALUE.SYS_REGMAP_SIZE PARAM_VALUE.SYS_REGMAP_SIZE } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.SYS_REGMAP_SIZE}] ${MODELPARAM_VALUE.SYS_REGMAP_SIZE}
}

proc update_MODELPARAM_VALUE.SYS_REGMAP_VER { MODELPARAM_VALUE.SYS_REGMAP_VER PARAM_VALUE.SYS_REGMAP_VER } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.SYS_REGMAP_VER}] ${MODELPARAM_VALUE.SYS_REGMAP_VER}
}

proc update_MODELPARAM_VALUE.TIME_YEAR { MODELPARAM_VALUE.TIME_YEAR PARAM_VALUE.TIME_YEAR } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.TIME_YEAR}] ${MODELPARAM_VALUE.TIME_YEAR}
}

proc update_MODELPARAM_VALUE.TIME_MONTH { MODELPARAM_VALUE.TIME_MONTH PARAM_VALUE.TIME_MONTH } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.TIME_MONTH}] ${MODELPARAM_VALUE.TIME_MONTH}
}

proc update_MODELPARAM_VALUE.TIME_DAY { MODELPARAM_VALUE.TIME_DAY PARAM_VALUE.TIME_DAY } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.TIME_DAY}] ${MODELPARAM_VALUE.TIME_DAY}
}

proc update_MODELPARAM_VALUE.TIME_HOUR { MODELPARAM_VALUE.TIME_HOUR PARAM_VALUE.TIME_HOUR } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.TIME_HOUR}] ${MODELPARAM_VALUE.TIME_HOUR}
}

proc update_MODELPARAM_VALUE.TIME_QUARTER { MODELPARAM_VALUE.TIME_QUARTER PARAM_VALUE.TIME_QUARTER } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.TIME_QUARTER}] ${MODELPARAM_VALUE.TIME_QUARTER}
}

proc update_MODELPARAM_VALUE.GIT_COMMIT_ID { MODELPARAM_VALUE.GIT_COMMIT_ID PARAM_VALUE.GIT_COMMIT_ID } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.GIT_COMMIT_ID}] ${MODELPARAM_VALUE.GIT_COMMIT_ID}
}

proc update_MODELPARAM_VALUE.GIT_COMMIT_TIME { MODELPARAM_VALUE.GIT_COMMIT_TIME PARAM_VALUE.GIT_COMMIT_TIME } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.GIT_COMMIT_TIME}] ${MODELPARAM_VALUE.GIT_COMMIT_TIME}
}

proc update_MODELPARAM_VALUE.VER_IP_REV { MODELPARAM_VALUE.VER_IP_REV PARAM_VALUE.VER_IP_REV } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.VER_IP_REV}] ${MODELPARAM_VALUE.VER_IP_REV}
}

proc update_MODELPARAM_VALUE.VER_TARGET { MODELPARAM_VALUE.VER_TARGET PARAM_VALUE.VER_TARGET } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.VER_TARGET}] ${MODELPARAM_VALUE.VER_TARGET}
}

proc update_MODELPARAM_VALUE.ARCH_HP_BW { MODELPARAM_VALUE.ARCH_HP_BW PARAM_VALUE.ARCH_HP_BW } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.ARCH_HP_BW}] ${MODELPARAM_VALUE.ARCH_HP_BW}
}

proc update_MODELPARAM_VALUE.ARCH_DATA_BW { MODELPARAM_VALUE.ARCH_DATA_BW PARAM_VALUE.ARCH_DATA_BW } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.ARCH_DATA_BW}] ${MODELPARAM_VALUE.ARCH_DATA_BW}
}

proc update_MODELPARAM_VALUE.ARCH_IMG_BKGRP { MODELPARAM_VALUE.ARCH_IMG_BKGRP PARAM_VALUE.ARCH_IMG_BKGRP } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.ARCH_IMG_BKGRP}] ${MODELPARAM_VALUE.ARCH_IMG_BKGRP}
}

proc update_MODELPARAM_VALUE.ARCH_PP { MODELPARAM_VALUE.ARCH_PP PARAM_VALUE.ARCH_PP } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.ARCH_PP}] ${MODELPARAM_VALUE.ARCH_PP}
}

proc update_MODELPARAM_VALUE.ARCH_ICP { MODELPARAM_VALUE.ARCH_ICP PARAM_VALUE.ARCH_ICP } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.ARCH_ICP}] ${MODELPARAM_VALUE.ARCH_ICP}
}

proc update_MODELPARAM_VALUE.ARCH_OCP { MODELPARAM_VALUE.ARCH_OCP PARAM_VALUE.ARCH_OCP } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.ARCH_OCP}] ${MODELPARAM_VALUE.ARCH_OCP}
}

proc update_MODELPARAM_VALUE.RAM_DEPTH_MEAN { MODELPARAM_VALUE.RAM_DEPTH_MEAN PARAM_VALUE.RAM_DEPTH_MEAN } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.RAM_DEPTH_MEAN}] ${MODELPARAM_VALUE.RAM_DEPTH_MEAN}
}

proc update_MODELPARAM_VALUE.RAM_DEPTH_BIAS { MODELPARAM_VALUE.RAM_DEPTH_BIAS PARAM_VALUE.RAM_DEPTH_BIAS } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.RAM_DEPTH_BIAS}] ${MODELPARAM_VALUE.RAM_DEPTH_BIAS}
}

proc update_MODELPARAM_VALUE.RAM_DEPTH_WGT { MODELPARAM_VALUE.RAM_DEPTH_WGT PARAM_VALUE.RAM_DEPTH_WGT } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.RAM_DEPTH_WGT}] ${MODELPARAM_VALUE.RAM_DEPTH_WGT}
}

proc update_MODELPARAM_VALUE.RAM_DEPTH_IMG { MODELPARAM_VALUE.RAM_DEPTH_IMG PARAM_VALUE.RAM_DEPTH_IMG } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.RAM_DEPTH_IMG}] ${MODELPARAM_VALUE.RAM_DEPTH_IMG}
}

proc update_MODELPARAM_VALUE.UBANK_IMG_N { MODELPARAM_VALUE.UBANK_IMG_N PARAM_VALUE.UBANK_IMG_N } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.UBANK_IMG_N}] ${MODELPARAM_VALUE.UBANK_IMG_N}
}

proc update_MODELPARAM_VALUE.UBANK_WGT_N { MODELPARAM_VALUE.UBANK_WGT_N PARAM_VALUE.UBANK_WGT_N } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.UBANK_WGT_N}] ${MODELPARAM_VALUE.UBANK_WGT_N}
}

proc update_MODELPARAM_VALUE.UBANK_BIAS { MODELPARAM_VALUE.UBANK_BIAS PARAM_VALUE.UBANK_BIAS } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.UBANK_BIAS}] ${MODELPARAM_VALUE.UBANK_BIAS}
}

proc update_MODELPARAM_VALUE.DBANK_IMG_N { MODELPARAM_VALUE.DBANK_IMG_N PARAM_VALUE.DBANK_IMG_N } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.DBANK_IMG_N}] ${MODELPARAM_VALUE.DBANK_IMG_N}
}

proc update_MODELPARAM_VALUE.DBANK_WGT_N { MODELPARAM_VALUE.DBANK_WGT_N PARAM_VALUE.DBANK_WGT_N } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.DBANK_WGT_N}] ${MODELPARAM_VALUE.DBANK_WGT_N}
}

proc update_MODELPARAM_VALUE.DBANK_BIAS { MODELPARAM_VALUE.DBANK_BIAS PARAM_VALUE.DBANK_BIAS } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.DBANK_BIAS}] ${MODELPARAM_VALUE.DBANK_BIAS}
}

proc update_MODELPARAM_VALUE.LOAD_AUGM { MODELPARAM_VALUE.LOAD_AUGM PARAM_VALUE.LOAD_AUGM } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.LOAD_AUGM}] ${MODELPARAM_VALUE.LOAD_AUGM}
}

proc update_MODELPARAM_VALUE.LOAD_IMG_MEAN { MODELPARAM_VALUE.LOAD_IMG_MEAN PARAM_VALUE.LOAD_IMG_MEAN } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.LOAD_IMG_MEAN}] ${MODELPARAM_VALUE.LOAD_IMG_MEAN}
}

proc update_MODELPARAM_VALUE.LOAD_PARALLEL { MODELPARAM_VALUE.LOAD_PARALLEL PARAM_VALUE.LOAD_PARALLEL } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.LOAD_PARALLEL}] ${MODELPARAM_VALUE.LOAD_PARALLEL}
}

proc update_MODELPARAM_VALUE.CONV_LEAKYRELU { MODELPARAM_VALUE.CONV_LEAKYRELU PARAM_VALUE.CONV_LEAKYRELU } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.CONV_LEAKYRELU}] ${MODELPARAM_VALUE.CONV_LEAKYRELU}
}

proc update_MODELPARAM_VALUE.CONV_RELU6 { MODELPARAM_VALUE.CONV_RELU6 PARAM_VALUE.CONV_RELU6 } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.CONV_RELU6}] ${MODELPARAM_VALUE.CONV_RELU6}
}

proc update_MODELPARAM_VALUE.CONV_WR_PARALLEL { MODELPARAM_VALUE.CONV_WR_PARALLEL PARAM_VALUE.CONV_WR_PARALLEL } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.CONV_WR_PARALLEL}] ${MODELPARAM_VALUE.CONV_WR_PARALLEL}
}

proc update_MODELPARAM_VALUE.CONV_DSP_CASC_MAX { MODELPARAM_VALUE.CONV_DSP_CASC_MAX PARAM_VALUE.CONV_DSP_CASC_MAX } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.CONV_DSP_CASC_MAX}] ${MODELPARAM_VALUE.CONV_DSP_CASC_MAX}
}

proc update_MODELPARAM_VALUE.CONV_DSP_ACCU_ENA { MODELPARAM_VALUE.CONV_DSP_ACCU_ENA PARAM_VALUE.CONV_DSP_ACCU_ENA } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.CONV_DSP_ACCU_ENA}] ${MODELPARAM_VALUE.CONV_DSP_ACCU_ENA}
}

proc update_MODELPARAM_VALUE.SAVE_PARALLEL { MODELPARAM_VALUE.SAVE_PARALLEL PARAM_VALUE.SAVE_PARALLEL } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.SAVE_PARALLEL}] ${MODELPARAM_VALUE.SAVE_PARALLEL}
}

proc update_MODELPARAM_VALUE.POOL_AVERAGE { MODELPARAM_VALUE.POOL_AVERAGE PARAM_VALUE.POOL_AVERAGE } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.POOL_AVERAGE}] ${MODELPARAM_VALUE.POOL_AVERAGE}
}

proc update_MODELPARAM_VALUE.ELEW_PARALLEL { MODELPARAM_VALUE.ELEW_PARALLEL PARAM_VALUE.ELEW_PARALLEL } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.ELEW_PARALLEL}] ${MODELPARAM_VALUE.ELEW_PARALLEL}
}

proc update_MODELPARAM_VALUE.ELEW_MULT_EN { MODELPARAM_VALUE.ELEW_MULT_EN PARAM_VALUE.ELEW_MULT_EN } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.ELEW_MULT_EN}] ${MODELPARAM_VALUE.ELEW_MULT_EN}
}

proc update_MODELPARAM_VALUE.DWCV_ALU_MODE { MODELPARAM_VALUE.DWCV_ALU_MODE PARAM_VALUE.DWCV_ALU_MODE } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.DWCV_ALU_MODE}] ${MODELPARAM_VALUE.DWCV_ALU_MODE}
}

proc update_MODELPARAM_VALUE.DWCV_RELU6 { MODELPARAM_VALUE.DWCV_RELU6 PARAM_VALUE.DWCV_RELU6 } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.DWCV_RELU6}] ${MODELPARAM_VALUE.DWCV_RELU6}
}

proc update_MODELPARAM_VALUE.DWCV_PARALLEL { MODELPARAM_VALUE.DWCV_PARALLEL PARAM_VALUE.DWCV_PARALLEL } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.DWCV_PARALLEL}] ${MODELPARAM_VALUE.DWCV_PARALLEL}
}

proc update_MODELPARAM_VALUE.MISC_WR_PARALLEL { MODELPARAM_VALUE.MISC_WR_PARALLEL PARAM_VALUE.MISC_WR_PARALLEL } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.MISC_WR_PARALLEL}] ${MODELPARAM_VALUE.MISC_WR_PARALLEL}
}

proc update_MODELPARAM_VALUE.DNNDK_PRINT { MODELPARAM_VALUE.DNNDK_PRINT PARAM_VALUE.DNNDK_PRINT } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.DNNDK_PRINT}] ${MODELPARAM_VALUE.DNNDK_PRINT}
}

proc update_MODELPARAM_VALUE.DPU1_GP_ID_BW { MODELPARAM_VALUE.DPU1_GP_ID_BW PARAM_VALUE.DPU1_GP_ID_BW } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.DPU1_GP_ID_BW}] ${MODELPARAM_VALUE.DPU1_GP_ID_BW}
}

proc update_MODELPARAM_VALUE.DPU1_HP0_ID_BW { MODELPARAM_VALUE.DPU1_HP0_ID_BW PARAM_VALUE.DPU1_HP0_ID_BW } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.DPU1_HP0_ID_BW}] ${MODELPARAM_VALUE.DPU1_HP0_ID_BW}
}

proc update_MODELPARAM_VALUE.DPU1_HP1_ID_BW { MODELPARAM_VALUE.DPU1_HP1_ID_BW PARAM_VALUE.DPU1_HP1_ID_BW } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.DPU1_HP1_ID_BW}] ${MODELPARAM_VALUE.DPU1_HP1_ID_BW}
}

proc update_MODELPARAM_VALUE.DPU1_HP2_ID_BW { MODELPARAM_VALUE.DPU1_HP2_ID_BW PARAM_VALUE.DPU1_HP2_ID_BW } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.DPU1_HP2_ID_BW}] ${MODELPARAM_VALUE.DPU1_HP2_ID_BW}
}

proc update_MODELPARAM_VALUE.DPU1_HP3_ID_BW { MODELPARAM_VALUE.DPU1_HP3_ID_BW PARAM_VALUE.DPU1_HP3_ID_BW } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.DPU1_HP3_ID_BW}] ${MODELPARAM_VALUE.DPU1_HP3_ID_BW}
}

proc update_MODELPARAM_VALUE.DPU1_UBANK_IMG_N { MODELPARAM_VALUE.DPU1_UBANK_IMG_N PARAM_VALUE.DPU1_UBANK_IMG_N } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.DPU1_UBANK_IMG_N}] ${MODELPARAM_VALUE.DPU1_UBANK_IMG_N}
}

proc update_MODELPARAM_VALUE.DPU1_UBANK_WGT_N { MODELPARAM_VALUE.DPU1_UBANK_WGT_N PARAM_VALUE.DPU1_UBANK_WGT_N } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.DPU1_UBANK_WGT_N}] ${MODELPARAM_VALUE.DPU1_UBANK_WGT_N}
}

proc update_MODELPARAM_VALUE.DPU1_UBANK_BIAS { MODELPARAM_VALUE.DPU1_UBANK_BIAS PARAM_VALUE.DPU1_UBANK_BIAS } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.DPU1_UBANK_BIAS}] ${MODELPARAM_VALUE.DPU1_UBANK_BIAS}
}

proc update_MODELPARAM_VALUE.DPU1_DBANK_IMG_N { MODELPARAM_VALUE.DPU1_DBANK_IMG_N PARAM_VALUE.DPU1_DBANK_IMG_N } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.DPU1_DBANK_IMG_N}] ${MODELPARAM_VALUE.DPU1_DBANK_IMG_N}
}

proc update_MODELPARAM_VALUE.DPU1_DBANK_WGT_N { MODELPARAM_VALUE.DPU1_DBANK_WGT_N PARAM_VALUE.DPU1_DBANK_WGT_N } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.DPU1_DBANK_WGT_N}] ${MODELPARAM_VALUE.DPU1_DBANK_WGT_N}
}

proc update_MODELPARAM_VALUE.DPU1_DBANK_BIAS { MODELPARAM_VALUE.DPU1_DBANK_BIAS PARAM_VALUE.DPU1_DBANK_BIAS } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.DPU1_DBANK_BIAS}] ${MODELPARAM_VALUE.DPU1_DBANK_BIAS}
}

proc update_MODELPARAM_VALUE.DPU2_GP_ID_BW { MODELPARAM_VALUE.DPU2_GP_ID_BW PARAM_VALUE.DPU2_GP_ID_BW } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.DPU2_GP_ID_BW}] ${MODELPARAM_VALUE.DPU2_GP_ID_BW}
}

proc update_MODELPARAM_VALUE.DPU2_HP0_ID_BW { MODELPARAM_VALUE.DPU2_HP0_ID_BW PARAM_VALUE.DPU2_HP0_ID_BW } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.DPU2_HP0_ID_BW}] ${MODELPARAM_VALUE.DPU2_HP0_ID_BW}
}

proc update_MODELPARAM_VALUE.DPU2_HP1_ID_BW { MODELPARAM_VALUE.DPU2_HP1_ID_BW PARAM_VALUE.DPU2_HP1_ID_BW } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.DPU2_HP1_ID_BW}] ${MODELPARAM_VALUE.DPU2_HP1_ID_BW}
}

proc update_MODELPARAM_VALUE.DPU2_HP2_ID_BW { MODELPARAM_VALUE.DPU2_HP2_ID_BW PARAM_VALUE.DPU2_HP2_ID_BW } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.DPU2_HP2_ID_BW}] ${MODELPARAM_VALUE.DPU2_HP2_ID_BW}
}

proc update_MODELPARAM_VALUE.DPU2_HP3_ID_BW { MODELPARAM_VALUE.DPU2_HP3_ID_BW PARAM_VALUE.DPU2_HP3_ID_BW } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.DPU2_HP3_ID_BW}] ${MODELPARAM_VALUE.DPU2_HP3_ID_BW}
}

proc update_MODELPARAM_VALUE.DPU2_UBANK_IMG_N { MODELPARAM_VALUE.DPU2_UBANK_IMG_N PARAM_VALUE.DPU2_UBANK_IMG_N } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.DPU2_UBANK_IMG_N}] ${MODELPARAM_VALUE.DPU2_UBANK_IMG_N}
}

proc update_MODELPARAM_VALUE.DPU2_UBANK_WGT_N { MODELPARAM_VALUE.DPU2_UBANK_WGT_N PARAM_VALUE.DPU2_UBANK_WGT_N } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.DPU2_UBANK_WGT_N}] ${MODELPARAM_VALUE.DPU2_UBANK_WGT_N}
}

proc update_MODELPARAM_VALUE.DPU2_UBANK_BIAS { MODELPARAM_VALUE.DPU2_UBANK_BIAS PARAM_VALUE.DPU2_UBANK_BIAS } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.DPU2_UBANK_BIAS}] ${MODELPARAM_VALUE.DPU2_UBANK_BIAS}
}

proc update_MODELPARAM_VALUE.DPU2_DBANK_IMG_N { MODELPARAM_VALUE.DPU2_DBANK_IMG_N PARAM_VALUE.DPU2_DBANK_IMG_N } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.DPU2_DBANK_IMG_N}] ${MODELPARAM_VALUE.DPU2_DBANK_IMG_N}
}

proc update_MODELPARAM_VALUE.DPU2_DBANK_WGT_N { MODELPARAM_VALUE.DPU2_DBANK_WGT_N PARAM_VALUE.DPU2_DBANK_WGT_N } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.DPU2_DBANK_WGT_N}] ${MODELPARAM_VALUE.DPU2_DBANK_WGT_N}
}

proc update_MODELPARAM_VALUE.DPU2_DBANK_BIAS { MODELPARAM_VALUE.DPU2_DBANK_BIAS PARAM_VALUE.DPU2_DBANK_BIAS } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.DPU2_DBANK_BIAS}] ${MODELPARAM_VALUE.DPU2_DBANK_BIAS}
}

proc update_MODELPARAM_VALUE.DPU3_GP_ID_BW { MODELPARAM_VALUE.DPU3_GP_ID_BW PARAM_VALUE.DPU3_GP_ID_BW } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.DPU3_GP_ID_BW}] ${MODELPARAM_VALUE.DPU3_GP_ID_BW}
}

proc update_MODELPARAM_VALUE.DPU3_HP0_ID_BW { MODELPARAM_VALUE.DPU3_HP0_ID_BW PARAM_VALUE.DPU3_HP0_ID_BW } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.DPU3_HP0_ID_BW}] ${MODELPARAM_VALUE.DPU3_HP0_ID_BW}
}

proc update_MODELPARAM_VALUE.DPU3_HP1_ID_BW { MODELPARAM_VALUE.DPU3_HP1_ID_BW PARAM_VALUE.DPU3_HP1_ID_BW } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.DPU3_HP1_ID_BW}] ${MODELPARAM_VALUE.DPU3_HP1_ID_BW}
}

proc update_MODELPARAM_VALUE.DPU3_HP2_ID_BW { MODELPARAM_VALUE.DPU3_HP2_ID_BW PARAM_VALUE.DPU3_HP2_ID_BW } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.DPU3_HP2_ID_BW}] ${MODELPARAM_VALUE.DPU3_HP2_ID_BW}
}

proc update_MODELPARAM_VALUE.DPU3_HP3_ID_BW { MODELPARAM_VALUE.DPU3_HP3_ID_BW PARAM_VALUE.DPU3_HP3_ID_BW } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.DPU3_HP3_ID_BW}] ${MODELPARAM_VALUE.DPU3_HP3_ID_BW}
}

proc update_MODELPARAM_VALUE.DPU3_UBANK_IMG_N { MODELPARAM_VALUE.DPU3_UBANK_IMG_N PARAM_VALUE.DPU3_UBANK_IMG_N } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.DPU3_UBANK_IMG_N}] ${MODELPARAM_VALUE.DPU3_UBANK_IMG_N}
}

proc update_MODELPARAM_VALUE.DPU3_UBANK_WGT_N { MODELPARAM_VALUE.DPU3_UBANK_WGT_N PARAM_VALUE.DPU3_UBANK_WGT_N } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.DPU3_UBANK_WGT_N}] ${MODELPARAM_VALUE.DPU3_UBANK_WGT_N}
}

proc update_MODELPARAM_VALUE.DPU3_UBANK_BIAS { MODELPARAM_VALUE.DPU3_UBANK_BIAS PARAM_VALUE.DPU3_UBANK_BIAS } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.DPU3_UBANK_BIAS}] ${MODELPARAM_VALUE.DPU3_UBANK_BIAS}
}

proc update_MODELPARAM_VALUE.DPU3_DBANK_IMG_N { MODELPARAM_VALUE.DPU3_DBANK_IMG_N PARAM_VALUE.DPU3_DBANK_IMG_N } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.DPU3_DBANK_IMG_N}] ${MODELPARAM_VALUE.DPU3_DBANK_IMG_N}
}

proc update_MODELPARAM_VALUE.DPU3_DBANK_WGT_N { MODELPARAM_VALUE.DPU3_DBANK_WGT_N PARAM_VALUE.DPU3_DBANK_WGT_N } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.DPU3_DBANK_WGT_N}] ${MODELPARAM_VALUE.DPU3_DBANK_WGT_N}
}

proc update_MODELPARAM_VALUE.DPU3_DBANK_BIAS { MODELPARAM_VALUE.DPU3_DBANK_BIAS PARAM_VALUE.DPU3_DBANK_BIAS } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.DPU3_DBANK_BIAS}] ${MODELPARAM_VALUE.DPU3_DBANK_BIAS}
}

proc update_MODELPARAM_VALUE.SFM_ENA { MODELPARAM_VALUE.SFM_ENA PARAM_VALUE.SFM_ENA } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.SFM_ENA}] ${MODELPARAM_VALUE.SFM_ENA}
}

proc update_MODELPARAM_VALUE.SFM_HP0_ID_BW { MODELPARAM_VALUE.SFM_HP0_ID_BW PARAM_VALUE.SFM_HP0_ID_BW } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.SFM_HP0_ID_BW}] ${MODELPARAM_VALUE.SFM_HP0_ID_BW}
}

proc update_MODELPARAM_VALUE.SFM_HP_DATA_BW { MODELPARAM_VALUE.SFM_HP_DATA_BW PARAM_VALUE.SFM_HP_DATA_BW } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.SFM_HP_DATA_BW}] ${MODELPARAM_VALUE.SFM_HP_DATA_BW}
}

