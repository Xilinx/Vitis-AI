#CSR constraints
#Read-Only CSR
set_false_path -to [get_pins "m_regmap/int_dpu0_status00_Q0_reg*/D"]
set_false_path -to [get_pins "m_regmap/int_dpu0_status01_reg*/D"]
set_false_path -to [get_pins "m_regmap/int_dpu0_status02_reg*/D"]
set_false_path -to [get_pins "m_regmap/int_dpu0_status03_reg*/D"]
set_false_path -to [get_pins "m_regmap/int_dpu0_status04_reg*/D"]
set_false_path -to [get_pins "m_regmap/int_dpu0_status05_reg*/D"]
set_false_path -to [get_pins "m_regmap/int_dpu0_status06_reg*/D"]
set_false_path -to [get_pins "m_regmap/int_dpu0_status07_reg*/D"]
set_false_path -to [get_pins "m_regmap/int_dpu0_status08_reg*/D"]
#set_false_path -to [get_pins "m_regmap/int_dpu0_status09_reg*/D"]
set_false_path -to [get_pins "m_regmap/int_dpu0_status10_reg*/D"]

#Read-Write CSR
set_false_path -from [get_pins "m_regmap/int_dpu0_scalar00_reg*/C"]
#set_false_path -from [get_pins "m_regmap/int_dpu0_scalar01_reg*/C"]
set_false_path -from [get_pins "m_regmap/int_dpu0_scalar02_reg*/C"]
set_false_path -from [get_pins "m_regmap/int_dpu0_scalar03_reg*/C"]
set_false_path -from [get_pins "m_regmap/int_dpu0_scalar04_reg*/C"]
set_false_path -from [get_pins "m_regmap/int_dpu0_scalar05_reg*/C"]
set_false_path -from [get_pins "m_regmap/int_dpu0_scalar06_reg*/C"]
set_false_path -from [get_pins "m_regmap/int_dpu0_scalar07_reg*/C"]
set_false_path -from [get_pins "m_regmap/int_dpu0_scalar08_reg*/C"]
set_false_path -from [get_pins "m_regmap/int_dpu0_scalar09_reg*/C"]
set_false_path -from [get_pins "m_regmap/int_dpu0_scalar10_reg*/C"]
set_false_path -from [get_pins "m_regmap/int_dpu0_scalar11_reg*/C"]
set_false_path -from [get_pins "m_regmap/int_dpu0_scalar12_reg*/C"]

set_false_path -from [get_pins "m_regmap/int_dpu0_axi00_ptr00_reg*/C"]
set_false_path -from [get_pins "m_regmap/int_dpu0_axi00_ptr01_reg*/C"]
set_false_path -from [get_pins "m_regmap/int_dpu0_axi00_ptr02_reg*/C"]
set_false_path -from [get_pins "m_regmap/int_dpu0_axi00_ptr03_reg*/C"]
set_false_path -from [get_pins "m_regmap/int_dpu0_axi00_ptr04_reg*/C"]
set_false_path -from [get_pins "m_regmap/int_dpu0_axi00_ptr05_reg*/C"]
set_false_path -from [get_pins "m_regmap/int_dpu0_axi00_ptr06_reg*/C"]
set_false_path -from [get_pins "m_regmap/int_dpu0_axi00_ptr07_reg*/C"]
set_false_path -from [get_pins "m_regmap/int_dpu0_axi00_ptr08_reg*/C"]
#set_false_path -from [get_pins "m_regmap/int_dpu0_axi00_ptr09_reg*/C"]

#2019-03-26, add false path for dpu_interrupt
set_false_path -thr [get_ports -scoped_to_current_instance "dpu_interrupt"]

#2019-05-17, add false path to ap_start
set_false_path -to [get_pins "m_regmap/dpu0_scalar00_Q_reg*/D"]
#2019-05-17, add false path to finish_clr
set_false_path -to [get_pins "m_regmap/dpu0_scalar02_Q_reg*/D"]

#Conv MCP
#2018-11-09, from Pengbo, add MCP from 666 to 333
#Bo, 2019-02-21, remove below path from *tmp* to *tmp* because of Pengbo's design update
#set_multicycle_path -setup 2 -start \
#-from [get_pins "m_conv_top/inst_pea/GenOcp[*].u_pe/GenMultCvb.GenMultech[*].GenLstPeDataGrp[*].add_res_last_tmp_reg_0_reg[*]/C"] \
#-to   [get_pins "m_conv_top/inst_pea/GenOcp[*].u_pe/GenMultCvb.GenMultech[*].GenLstPeDataGrp[*].add_res_last_tmp_rr_0_reg[*]/D"]
#set_multicycle_path -hold 1 -start \
#-from [get_pins "m_conv_top/inst_pea/GenOcp[*].u_pe/GenMultCvb.GenMultech[*].GenLstPeDataGrp[*].add_res_last_tmp_reg_0_reg[*]/C"] \
#-to   [get_pins "m_conv_top/inst_pea/GenOcp[*].u_pe/GenMultCvb.GenMultech[*].GenLstPeDataGrp[*].add_res_last_tmp_rr_0_reg[*]/D"]
#
##set_multicycle_path -setup 2 -start \
#-from [get_pins "m_conv_top/inst_pea/GenOcp[*].u_pe/GenMultCvb.GenMultech[*].GenLstPeDataGrp[*].add_res_last_tmp_reg_0_reg[*]/C"] \
#-to   [get_pins "m_conv_top/inst_pea/GenOcp[*].u_pe/GenMultCvb.GenMultech[*].GenLstPeDataGrp[*].add_res_last_tmp_rr_0_reg[*]/D"]
##set_multicycle_path -hold 1 -start \
#-from [get_pins "m_conv_top/inst_pea/GenOcp[*].u_pe/GenMultCvb.GenMultech[*].GenLstPeDataGrp[*].add_res_last_tmp_reg_0_reg[*]/C"] \
#-to   [get_pins "m_conv_top/inst_pea/GenOcp[*].u_pe/GenMultCvb.GenMultech[*].GenLstPeDataGrp[*].add_res_last_tmp_rr_0_reg[*]/D"]
#
#set_multicycle_path -setup 2 -start \
#-from [get_pins "m_conv_top/inst_pea/GenOcp[*].u_pe/GenMultCvb.GenMultech[*].GenLstPeDataGrp[*].add_res_last_tmp_reg_1_reg[*]/C"] \
#-to   [get_pins "m_conv_top/inst_pea/GenOcp[*].u_pe/GenMultCvb.GenMultech[*].GenLstPeDataGrp[*].add_res_last_tmp_rr_1_reg[*]/D"]
#set_multicycle_path -hold 1 -start \
#-from [get_pins "m_conv_top/inst_pea/GenOcp[*].u_pe/GenMultCvb.GenMultech[*].GenLstPeDataGrp[*].add_res_last_tmp_reg_1_reg[*]/C"] \
#-to   [get_pins "m_conv_top/inst_pea/GenOcp[*].u_pe/GenMultCvb.GenMultech[*].GenLstPeDataGrp[*].add_res_last_tmp_rr_1_reg[*]/D"]
#
##set_multicycle_path -hold 1 -start \
#-from [get_pins "m_conv_top/inst_pea/GenOcp[*].u_pe/GenMultCvb.GenMultech[*].GenLstPeDataGrp[*].add_res_last_tmp_reg_1_reg[*]/C"] \
#-to   [get_pins "m_conv_top/inst_pea/GenOcp[*].u_pe/GenMultCvb.GenMultech[*].GenLstPeDataGrp[*].add_res_last_tmp_rr_1_reg[*]/D"]




set_multicycle_path -setup 2 -start \
-from [get_pins "m_conv_top/inst_pea/GenOcp[*].u_pe/GenMultCvb.GenMultech[*].GenLstPeDataGrp[*].add_res_8_last_reg_0_reg[*]/C"] \
-to   [get_pins "m_conv_top/inst_pea/GenOcp[*].u_pe/GenMultCvb.GenMultech[*].GenLstPeDataGrp[*].add_res_8_last_rr_0_reg[*]/D"]
set_multicycle_path -hold 1 -start \
-from [get_pins "m_conv_top/inst_pea/GenOcp[*].u_pe/GenMultCvb.GenMultech[*].GenLstPeDataGrp[*].add_res_8_last_reg_0_reg[*]/C"] \
-to   [get_pins "m_conv_top/inst_pea/GenOcp[*].u_pe/GenMultCvb.GenMultech[*].GenLstPeDataGrp[*].add_res_8_last_rr_0_reg[*]/D"]

#set_multicycle_path -setup 2 -start \
#-from [get_pins "m_conv_top/inst_pea/GenOcp[*].u_pe/GenMultCvb.GenMultech[*].GenLstPeDataGrp[*].add_res_8_last_reg_0_reg[*]/C"] \
#-to   [get_pins "m_conv_top/inst_pea/GenOcp[*].u_pe/GenMultCvb.GenMultech[*].GenLstPeDataGrp[*].add_res_8_last_rr_0_reg[*]/D"]
#set_multicycle_path -hold 1 -start \
#-from [get_pins "m_conv_top/inst_pea/GenOcp[*].u_pe/GenMultCvb.GenMultech[*].GenLstPeDataGrp[*].add_res_8_last_reg_0_reg[*]/C"] \
#-to   [get_pins "m_conv_top/inst_pea/GenOcp[*].u_pe/GenMultCvb.GenMultech[*].GenLstPeDataGrp[*].add_res_8_last_rr_0_reg[*]/D"]

set_multicycle_path -setup 2 -start \
-from [get_pins "m_conv_top/inst_pea/GenOcp[*].u_pe/GenMultCvb.GenMultech[*].GenLstPeDataGrp[*].add_res_8_last_reg_1_reg[*]/C"] \
-to   [get_pins "m_conv_top/inst_pea/GenOcp[*].u_pe/GenMultCvb.GenMultech[*].GenLstPeDataGrp[*].add_res_8_last_rr_1_reg[*]/D"]
set_multicycle_path -hold 1 -start \
-from [get_pins "m_conv_top/inst_pea/GenOcp[*].u_pe/GenMultCvb.GenMultech[*].GenLstPeDataGrp[*].add_res_8_last_reg_1_reg[*]/C"] \
-to   [get_pins "m_conv_top/inst_pea/GenOcp[*].u_pe/GenMultCvb.GenMultech[*].GenLstPeDataGrp[*].add_res_8_last_rr_1_reg[*]/D"]

#set_multicycle_path -hold 1 -start \
#-from [get_pins "m_conv_top/inst_pea/GenOcp[*].u_pe/GenMultCvb.GenMultech[*].GenLstPeDataGrp[*].add_res_8_last_reg_1_reg[*]/C"] \
#-to   [get_pins "m_conv_top/inst_pea/GenOcp[*].u_pe/GenMultCvb.GenMultech[*].GenLstPeDataGrp[*].add_res_8_last_rr_1_reg[*]/D"]

#From 333 to 666
#chain0
set_multicycle_path -setup 2 -end \
-from [get_pins "m_conv_top/inst_pea/GenOcp[*].u_pe/GenMultCvb.GenMultech[*].u_conv_block/GenConvChain[0].inst_dly_tdata_01/d_r_f_reg[*]/C"] \
-to   [get_pins "m_conv_top/inst_pea/GenOcp[*].u_pe/GenMultCvb.GenMultech[*].u_conv_block/GenConvChain[0].GenConvPair[*].Gen8x8.GenMultDsp.GenFirDsp.u_dsp_unit_fir/DSP48E2_gen.DSP48E2_inst/DSP_PREADD_DATA_INST/DIN[*]"]
set_multicycle_path -hold 1 -end \
-from [get_pins "m_conv_top/inst_pea/GenOcp[*].u_pe/GenMultCvb.GenMultech[*].u_conv_block/GenConvChain[0].inst_dly_tdata_01/d_r_f_reg[*]/C"] \
-to   [get_pins "m_conv_top/inst_pea/GenOcp[*].u_pe/GenMultCvb.GenMultech[*].u_conv_block/GenConvChain[0].GenConvPair[*].Gen8x8.GenMultDsp.GenFirDsp.u_dsp_unit_fir/DSP48E2_gen.DSP48E2_inst/DSP_PREADD_DATA_INST/DIN[*]"]

set_multicycle_path -setup 2 -end \
-from [get_pins "m_conv_top/inst_pea/GenOcp[*].u_pe/GenMultCvb.GenMultech[*].u_conv_block/GenConvChain[0].inst_dly_tdata_01/d_r_f_reg[*]/C"] \
-to   [get_pins "m_conv_top/inst_pea/GenOcp[*].u_pe/GenMultCvb.GenMultech[*].u_conv_block/GenConvChain[0].GenConvPair[*].Gen8x8.GenMultDsp.GenFirDsp.u_dsp_unit_fir/DSP48E2_gen.DSP48E2_inst/DSP_A_B_DATA_INST/A[*]"]
set_multicycle_path -hold 1 -end \
-from [get_pins "m_conv_top/inst_pea/GenOcp[*].u_pe/GenMultCvb.GenMultech[*].u_conv_block/GenConvChain[0].inst_dly_tdata_01/d_r_f_reg[*]/C"] \
-to   [get_pins "m_conv_top/inst_pea/GenOcp[*].u_pe/GenMultCvb.GenMultech[*].u_conv_block/GenConvChain[0].GenConvPair[*].Gen8x8.GenMultDsp.GenFirDsp.u_dsp_unit_fir/DSP48E2_gen.DSP48E2_inst/DSP_A_B_DATA_INST/A[*]"]

#chain 1
set_multicycle_path -setup 2 -end \
-from [get_pins "m_conv_top/inst_pea/GenOcp[*].u_pe/GenMultCvb.GenMultech[*].u_conv_block/GenConvChain[1].inst_dly_tdata_01/d_r_f_reg[*]/C"] \
-to   [get_pins "m_conv_top/inst_pea/GenOcp[*].u_pe/GenMultCvb.GenMultech[*].u_conv_block/GenConvChain[1].pe_tdata_grp_dly_odd_reg[*]/D"]
set_multicycle_path -hold 1 -end \
-from [get_pins "m_conv_top/inst_pea/GenOcp[*].u_pe/GenMultCvb.GenMultech[*].u_conv_block/GenConvChain[1].inst_dly_tdata_01/d_r_f_reg[*]/C"] \
-to   [get_pins "m_conv_top/inst_pea/GenOcp[*].u_pe/GenMultCvb.GenMultech[*].u_conv_block/GenConvChain[1].pe_tdata_grp_dly_odd_reg[*]/D"]

#chain2
set_multicycle_path -setup 2 -end \
-from [get_pins "m_conv_top/inst_pea/GenOcp[*].u_pe/GenMultCvb.GenMultech[*].u_conv_block/GenConvChain[2].inst_dly_tdata/d_r_f_reg[*]/C"] \
-to   [get_pins "m_conv_top/inst_pea/GenOcp[*].u_pe/GenMultCvb.GenMultech[*].u_conv_block/GenConvChain[2].GenConvPair[*].Gen8x8.GenMultDsp.GenMidDsp.u_dsp_unit_mid/DSP48E2_gen.DSP48E2_inst/DSP_PREADD_DATA_INST/DIN[*]"]
set_multicycle_path -hold 1 -end \
-from [get_pins "m_conv_top/inst_pea/GenOcp[*].u_pe/GenMultCvb.GenMultech[*].u_conv_block/GenConvChain[2].inst_dly_tdata/d_r_f_reg[*]/C"] \
-to   [get_pins "m_conv_top/inst_pea/GenOcp[*].u_pe/GenMultCvb.GenMultech[*].u_conv_block/GenConvChain[2].GenConvPair[*].Gen8x8.GenMultDsp.GenMidDsp.u_dsp_unit_mid/DSP48E2_gen.DSP48E2_inst/DSP_PREADD_DATA_INST/DIN[*]"]

set_multicycle_path -setup 2 -end \
-from [get_pins "m_conv_top/inst_pea/GenOcp[*].u_pe/GenMultCvb.GenMultech[*].u_conv_block/GenConvChain[2].inst_dly_tdata/d_r_f_reg[*]/C"] \
-to   [get_pins "m_conv_top/inst_pea/GenOcp[*].u_pe/GenMultCvb.GenMultech[*].u_conv_block/GenConvChain[2].GenConvPair[*].Gen8x8.GenMultDsp.GenMidDsp.u_dsp_unit_mid/DSP48E2_gen.DSP48E2_inst/DSP_A_B_DATA_INST/A[*]"]
set_multicycle_path -hold 1 -end \
-from [get_pins "m_conv_top/inst_pea/GenOcp[*].u_pe/GenMultCvb.GenMultech[*].u_conv_block/GenConvChain[2].inst_dly_tdata/d_r_f_reg[*]/C"] \
-to   [get_pins "m_conv_top/inst_pea/GenOcp[*].u_pe/GenMultCvb.GenMultech[*].u_conv_block/GenConvChain[2].GenConvPair[*].Gen8x8.GenMultDsp.GenMidDsp.u_dsp_unit_mid/DSP48E2_gen.DSP48E2_inst/DSP_A_B_DATA_INST/A[*]"]

#chain 3
set_multicycle_path -setup 2 -end \
-from [get_pins "m_conv_top/inst_pea/GenOcp[*].u_pe/GenMultCvb.GenMultech[*].u_conv_block/GenConvChain[3].inst_dly_tdata/d_r_f_reg[*]/C"] \
-to   [get_pins "m_conv_top/inst_pea/GenOcp[*].u_pe/GenMultCvb.GenMultech[*].u_conv_block/GenConvChain[3].pe_tdata_grp_dly_odd_reg[*]/D"]
set_multicycle_path -hold 1 -end \
-from [get_pins "m_conv_top/inst_pea/GenOcp[*].u_pe/GenMultCvb.GenMultech[*].u_conv_block/GenConvChain[3].inst_dly_tdata/d_r_f_reg[*]/C"] \
-to   [get_pins "m_conv_top/inst_pea/GenOcp[*].u_pe/GenMultCvb.GenMultech[*].u_conv_block/GenConvChain[3].pe_tdata_grp_dly_odd_reg[*]/D"]

#chain 4
set_multicycle_path -setup 2 -end \
-from [get_pins "m_conv_top/inst_pea/GenOcp[*].u_pe/GenMultCvb.GenMultech[*].u_conv_block/GenConvChain[4].inst_dly_tdata/d_r_f_reg[*]/C"] \
-to   [get_pins "m_conv_top/inst_pea/GenOcp[*].u_pe/GenMultCvb.GenMultech[*].u_conv_block/GenConvChain[4].GenConvPair[*].Gen8x8.GenMultDsp.GenMidDsp.u_dsp_unit_mid/DSP48E2_gen.DSP48E2_inst/DSP_PREADD_DATA_INST/DIN[*]"]
set_multicycle_path -hold 1 -end \
-from [get_pins "m_conv_top/inst_pea/GenOcp[*].u_pe/GenMultCvb.GenMultech[*].u_conv_block/GenConvChain[4].inst_dly_tdata/d_r_f_reg[*]/C"] \
-to   [get_pins "m_conv_top/inst_pea/GenOcp[*].u_pe/GenMultCvb.GenMultech[*].u_conv_block/GenConvChain[4].GenConvPair[*].Gen8x8.GenMultDsp.GenMidDsp.u_dsp_unit_mid/DSP48E2_gen.DSP48E2_inst/DSP_PREADD_DATA_INST/DIN[*]"]

set_multicycle_path -setup 2 -end \
-from [get_pins "m_conv_top/inst_pea/GenOcp[*].u_pe/GenMultCvb.GenMultech[*].u_conv_block/GenConvChain[4].inst_dly_tdata/d_r_f_reg[*]/C"] \
-to   [get_pins "m_conv_top/inst_pea/GenOcp[*].u_pe/GenMultCvb.GenMultech[*].u_conv_block/GenConvChain[4].GenConvPair[*].Gen8x8.GenMultDsp.GenMidDsp.u_dsp_unit_mid/DSP48E2_gen.DSP48E2_inst/DSP_A_B_DATA_INST/A[*]"]
set_multicycle_path -hold 1 -end \
-from [get_pins "m_conv_top/inst_pea/GenOcp[*].u_pe/GenMultCvb.GenMultech[*].u_conv_block/GenConvChain[4].inst_dly_tdata/d_r_f_reg[*]/C"] \
-to   [get_pins "m_conv_top/inst_pea/GenOcp[*].u_pe/GenMultCvb.GenMultech[*].u_conv_block/GenConvChain[4].GenConvPair[*].Gen8x8.GenMultDsp.GenMidDsp.u_dsp_unit_mid/DSP48E2_gen.DSP48E2_inst/DSP_A_B_DATA_INST/A[*]"]


#chain 5
set_multicycle_path -setup 2 -end \
-from [get_pins "m_conv_top/inst_pea/GenOcp[*].u_pe/GenMultCvb.GenMultech[*].u_conv_block/GenConvChain[5].inst_dly_tdata/d_r_f_reg[*]/C"] \
-to   [get_pins "m_conv_top/inst_pea/GenOcp[*].u_pe/GenMultCvb.GenMultech[*].u_conv_block/GenConvChain[5].pe_tdata_grp_dly_odd_reg[*]/D"]
set_multicycle_path -hold 1 -end \
-from [get_pins "m_conv_top/inst_pea/GenOcp[*].u_pe/GenMultCvb.GenMultech[*].u_conv_block/GenConvChain[5].inst_dly_tdata/d_r_f_reg[*]/C"] \
-to   [get_pins "m_conv_top/inst_pea/GenOcp[*].u_pe/GenMultCvb.GenMultech[*].u_conv_block/GenConvChain[5].pe_tdata_grp_dly_odd_reg[*]/D"]

#chain6
set_multicycle_path -setup 2 -end \
-from [get_pins "m_conv_top/inst_pea/GenOcp[*].u_pe/GenMultCvb.GenMultech[*].u_conv_block/GenConvChain[6].inst_dly_tdata/d_r_f_reg[*]/C"] \
-to   [get_pins "m_conv_top/inst_pea/GenOcp[*].u_pe/GenMultCvb.GenMultech[*].u_conv_block/GenConvChain[6].GenConvPair[*].Gen8x8.GenMultDsp.GenMidDsp.u_dsp_unit_mid/DSP48E2_gen.DSP48E2_inst/DSP_PREADD_DATA_INST/DIN[*]"]
set_multicycle_path -hold 1 -end \
-from [get_pins "m_conv_top/inst_pea/GenOcp[*].u_pe/GenMultCvb.GenMultech[*].u_conv_block/GenConvChain[6].inst_dly_tdata/d_r_f_reg[*]/C"] \
-to   [get_pins "m_conv_top/inst_pea/GenOcp[*].u_pe/GenMultCvb.GenMultech[*].u_conv_block/GenConvChain[6].GenConvPair[*].Gen8x8.GenMultDsp.GenMidDsp.u_dsp_unit_mid/DSP48E2_gen.DSP48E2_inst/DSP_PREADD_DATA_INST/DIN[*]"]

set_multicycle_path -setup 2 -end \
-from [get_pins "m_conv_top/inst_pea/GenOcp[*].u_pe/GenMultCvb.GenMultech[*].u_conv_block/GenConvChain[6].inst_dly_tdata/d_r_f_reg[*]/C"] \
-to   [get_pins "m_conv_top/inst_pea/GenOcp[*].u_pe/GenMultCvb.GenMultech[*].u_conv_block/GenConvChain[6].GenConvPair[*].Gen8x8.GenMultDsp.GenMidDsp.u_dsp_unit_mid/DSP48E2_gen.DSP48E2_inst/DSP_A_B_DATA_INST/A[*]"]
set_multicycle_path -hold 1 -end \
-from [get_pins "m_conv_top/inst_pea/GenOcp[*].u_pe/GenMultCvb.GenMultech[*].u_conv_block/GenConvChain[6].inst_dly_tdata/d_r_f_reg[*]/C"] \
-to   [get_pins "m_conv_top/inst_pea/GenOcp[*].u_pe/GenMultCvb.GenMultech[*].u_conv_block/GenConvChain[6].GenConvPair[*].Gen8x8.GenMultDsp.GenMidDsp.u_dsp_unit_mid/DSP48E2_gen.DSP48E2_inst/DSP_A_B_DATA_INST/A[*]"]


#chain 7
set_multicycle_path -setup 2 -end \
-from [get_pins "m_conv_top/inst_pea/GenOcp[*].u_pe/GenMultCvb.GenMultech[*].u_conv_block/GenConvChain[7].inst_dly_tdata/d_r_f_reg[*]/C"] \
-to   [get_pins "m_conv_top/inst_pea/GenOcp[*].u_pe/GenMultCvb.GenMultech[*].u_conv_block/GenConvChain[7].pe_tdata_grp_dly_odd_reg[*]/D"]
set_multicycle_path -hold 1 -end \
-from [get_pins "m_conv_top/inst_pea/GenOcp[*].u_pe/GenMultCvb.GenMultech[*].u_conv_block/GenConvChain[7].inst_dly_tdata/d_r_f_reg[*]/C"] \
-to   [get_pins "m_conv_top/inst_pea/GenOcp[*].u_pe/GenMultCvb.GenMultech[*].u_conv_block/GenConvChain[7].pe_tdata_grp_dly_odd_reg[*]/D"]


#Weights chain0
set_multicycle_path -setup 2 -end \
-from [get_pins "m_conv_top/inst_pea/GenOcp[*].u_pe/GenMultCvb.GenMultech[*].u_conv_block/GenConvChain[0].inst_dly_wgt_0/d_r_reg[0][*]*/C"] \
-to   [get_pins "m_conv_top/inst_pea/GenOcp[*].u_pe/GenMultCvb.GenMultech[*].u_conv_block/GenConvChain[0].b_dly_even_reg[*]/D"]
set_multicycle_path -hold 1 -end \
-from [get_pins "m_conv_top/inst_pea/GenOcp[*].u_pe/GenMultCvb.GenMultech[*].u_conv_block/GenConvChain[0].inst_dly_wgt_0/d_r_reg[0][*]*/C"] \
-to   [get_pins "m_conv_top/inst_pea/GenOcp[*].u_pe/GenMultCvb.GenMultech[*].u_conv_block/GenConvChain[0].b_dly_even_reg[*]/D"]
set_multicycle_path -setup 2 -end \
-from [get_pins "m_conv_top/inst_pea/GenOcp[*].u_pe/GenMultCvb.GenMultech[*].u_conv_block/GenConvChain[0].inst_dly_wgt_0/d_r_reg[0][*]*/C"] \
-to   [get_pins "m_conv_top/inst_pea/GenOcp[*].u_pe/GenMultCvb.GenMultech[*].u_conv_block/GenConvChain[0].b_dly_even_reg[*]/D"]
set_multicycle_path -hold 1 -end \
-from [get_pins "m_conv_top/inst_pea/GenOcp[*].u_pe/GenMultCvb.GenMultech[*].u_conv_block/GenConvChain[0].inst_dly_wgt_0/d_r_reg[0][*]*/C"] \
-to   [get_pins "m_conv_top/inst_pea/GenOcp[*].u_pe/GenMultCvb.GenMultech[*].u_conv_block/GenConvChain[0].b_dly_even_reg[*]/D"]

set_multicycle_path -setup 2 -end \
-from [get_pins "m_conv_top/inst_pea/GenOcp[*].u_pe/GenMultCvb.GenMultech[*].u_conv_block/GenConvChain[0].inst_dly_wgt_1/BIT[0].D0.d_r_reg[0][*]*/C"] \
-to   [get_pins "m_conv_top/inst_pea/GenOcp[*].u_pe/GenMultCvb.GenMultech[*].u_conv_block/GenConvChain[0].b_dly_even_reg[*]/D"]
set_multicycle_path -hold 1 -end \
-from [get_pins "m_conv_top/inst_pea/GenOcp[*].u_pe/GenMultCvb.GenMultech[*].u_conv_block/GenConvChain[0].inst_dly_wgt_1/BIT[0].D0.d_r_reg[0][*]*/C"] \
-to   [get_pins "m_conv_top/inst_pea/GenOcp[*].u_pe/GenMultCvb.GenMultech[*].u_conv_block/GenConvChain[0].b_dly_even_reg[*]/D"]
set_multicycle_path -setup 2 -end \
-from [get_pins "m_conv_top/inst_pea/GenOcp[*].u_pe/GenMultCvb.GenMultech[*].u_conv_block/GenConvChain[0].inst_dly_wgt_1/BIT[0].D0.d_r_reg[0][*]*/C"] \
-to   [get_pins "m_conv_top/inst_pea/GenOcp[*].u_pe/GenMultCvb.GenMultech[*].u_conv_block/GenConvChain[0].b_dly_even_reg[*]/D"]
set_multicycle_path -hold 1 -end \
-from [get_pins "m_conv_top/inst_pea/GenOcp[*].u_pe/GenMultCvb.GenMultech[*].u_conv_block/GenConvChain[0].inst_dly_wgt_1/BIT[0].D0.d_r_reg[0][*]*/C"] \
-to   [get_pins "m_conv_top/inst_pea/GenOcp[*].u_pe/GenMultCvb.GenMultech[*].u_conv_block/GenConvChain[0].b_dly_even_reg[*]/D"]

#Weights chain1
set_multicycle_path -setup 2 -end \
-from [get_pins "m_conv_top/inst_pea/GenOcp[*].u_pe/GenMultCvb.GenMultech[*].u_conv_block/GenConvChain[1].inst_dly_wgt_0/d_r_reg[0][*]*/C"] \
-to   [get_pins "m_conv_top/inst_pea/GenOcp[*].u_pe/GenMultCvb.GenMultech[*].u_conv_block/GenConvChain[1].b_dly_even_reg[*]/D"]
set_multicycle_path -hold 1 -end \
-from [get_pins "m_conv_top/inst_pea/GenOcp[*].u_pe/GenMultCvb.GenMultech[*].u_conv_block/GenConvChain[1].inst_dly_wgt_0/d_r_reg[0][*]*/C"] \
-to   [get_pins "m_conv_top/inst_pea/GenOcp[*].u_pe/GenMultCvb.GenMultech[*].u_conv_block/GenConvChain[1].b_dly_even_reg[*]/D"]
set_multicycle_path -setup 2 -end \
-from [get_pins "m_conv_top/inst_pea/GenOcp[*].u_pe/GenMultCvb.GenMultech[*].u_conv_block/GenConvChain[1].inst_dly_wgt_0/d_r_reg[0][*]*/C"] \
-to   [get_pins "m_conv_top/inst_pea/GenOcp[*].u_pe/GenMultCvb.GenMultech[*].u_conv_block/GenConvChain[1].b_dly_even_reg[*]/D"]
set_multicycle_path -hold 1 -end \
-from [get_pins "m_conv_top/inst_pea/GenOcp[*].u_pe/GenMultCvb.GenMultech[*].u_conv_block/GenConvChain[1].inst_dly_wgt_0/d_r_reg[0][*]*/C"] \
-to   [get_pins "m_conv_top/inst_pea/GenOcp[*].u_pe/GenMultCvb.GenMultech[*].u_conv_block/GenConvChain[1].b_dly_even_reg[*]/D"]

set_multicycle_path -setup 2 -end \
-from [get_pins "m_conv_top/inst_pea/GenOcp[*].u_pe/GenMultCvb.GenMultech[*].u_conv_block/GenConvChain[1].inst_dly_wgt_1/BIT[0].D0.d_r_reg[0][*]*/C"] \
-to   [get_pins "m_conv_top/inst_pea/GenOcp[*].u_pe/GenMultCvb.GenMultech[*].u_conv_block/GenConvChain[1].b_dly_even_reg[*]/D"]
set_multicycle_path -hold 1 -end \
-from [get_pins "m_conv_top/inst_pea/GenOcp[*].u_pe/GenMultCvb.GenMultech[*].u_conv_block/GenConvChain[1].inst_dly_wgt_1/BIT[0].D0.d_r_reg[0][*]*/C"] \
-to   [get_pins "m_conv_top/inst_pea/GenOcp[*].u_pe/GenMultCvb.GenMultech[*].u_conv_block/GenConvChain[1].b_dly_even_reg[*]/D"]
set_multicycle_path -setup 2 -end \
-from [get_pins "m_conv_top/inst_pea/GenOcp[*].u_pe/GenMultCvb.GenMultech[*].u_conv_block/GenConvChain[1].inst_dly_wgt_1/BIT[0].D0.d_r_reg[0][*]*/C"] \
-to   [get_pins "m_conv_top/inst_pea/GenOcp[*].u_pe/GenMultCvb.GenMultech[*].u_conv_block/GenConvChain[1].b_dly_even_reg[*]/D"]
set_multicycle_path -hold 1 -end \
-from [get_pins "m_conv_top/inst_pea/GenOcp[*].u_pe/GenMultCvb.GenMultech[*].u_conv_block/GenConvChain[1].inst_dly_wgt_1/BIT[0].D0.d_r_reg[0][*]*/C"] \
-to   [get_pins "m_conv_top/inst_pea/GenOcp[*].u_pe/GenMultCvb.GenMultech[*].u_conv_block/GenConvChain[1].b_dly_even_reg[*]/D"]

#Weights chain2
set_multicycle_path -setup 2 -end \
-from [get_pins "m_conv_top/inst_pea/GenOcp[*].u_pe/GenMultCvb.GenMultech[*].u_conv_block/GenConvChain[2].inst_dly_wgt_0/d_r_reg[1][*]*/C"] \
-to   [get_pins "m_conv_top/inst_pea/GenOcp[*].u_pe/GenMultCvb.GenMultech[*].u_conv_block/GenConvChain[2].b_dly_even_reg[*]/D"]
set_multicycle_path -hold 1 -end \
-from [get_pins "m_conv_top/inst_pea/GenOcp[*].u_pe/GenMultCvb.GenMultech[*].u_conv_block/GenConvChain[2].inst_dly_wgt_0/d_r_reg[1][*]*/C"] \
-to   [get_pins "m_conv_top/inst_pea/GenOcp[*].u_pe/GenMultCvb.GenMultech[*].u_conv_block/GenConvChain[2].b_dly_even_reg[*]/D"]
set_multicycle_path -setup 2 -end \
-from [get_pins "m_conv_top/inst_pea/GenOcp[*].u_pe/GenMultCvb.GenMultech[*].u_conv_block/GenConvChain[2].inst_dly_wgt_0/d_r_reg[1][*]*/C"] \
-to   [get_pins "m_conv_top/inst_pea/GenOcp[*].u_pe/GenMultCvb.GenMultech[*].u_conv_block/GenConvChain[2].b_dly_even_reg[*]/D"]
set_multicycle_path -hold 1 -end \
-from [get_pins "m_conv_top/inst_pea/GenOcp[*].u_pe/GenMultCvb.GenMultech[*].u_conv_block/GenConvChain[2].inst_dly_wgt_0/d_r_reg[1][*]*/C"] \
-to   [get_pins "m_conv_top/inst_pea/GenOcp[*].u_pe/GenMultCvb.GenMultech[*].u_conv_block/GenConvChain[2].b_dly_even_reg[*]/D"]

set_multicycle_path -setup 2 -end \
-from [get_pins "m_conv_top/inst_pea/GenOcp[*].u_pe/GenMultCvb.GenMultech[*].u_conv_block/GenConvChain[2].inst_dly_wgt_1/BIT[1].DX.d_r_reg[1][*]*/C"] \
-to   [get_pins "m_conv_top/inst_pea/GenOcp[*].u_pe/GenMultCvb.GenMultech[*].u_conv_block/GenConvChain[2].b_dly_even_reg[*]/D"]
set_multicycle_path -hold 1 -end \
-from [get_pins "m_conv_top/inst_pea/GenOcp[*].u_pe/GenMultCvb.GenMultech[*].u_conv_block/GenConvChain[2].inst_dly_wgt_1/BIT[1].DX.d_r_reg[1][*]*/C"] \
-to   [get_pins "m_conv_top/inst_pea/GenOcp[*].u_pe/GenMultCvb.GenMultech[*].u_conv_block/GenConvChain[2].b_dly_even_reg[*]/D"]
set_multicycle_path -setup 2 -end \
-from [get_pins "m_conv_top/inst_pea/GenOcp[*].u_pe/GenMultCvb.GenMultech[*].u_conv_block/GenConvChain[2].inst_dly_wgt_1/BIT[1].DX.d_r_reg[1][*]*/C"] \
-to   [get_pins "m_conv_top/inst_pea/GenOcp[*].u_pe/GenMultCvb.GenMultech[*].u_conv_block/GenConvChain[2].b_dly_even_reg[*]/D"]
set_multicycle_path -hold 1 -end \
-from [get_pins "m_conv_top/inst_pea/GenOcp[*].u_pe/GenMultCvb.GenMultech[*].u_conv_block/GenConvChain[2].inst_dly_wgt_1/BIT[1].DX.d_r_reg[1][*]*/C"] \
-to   [get_pins "m_conv_top/inst_pea/GenOcp[*].u_pe/GenMultCvb.GenMultech[*].u_conv_block/GenConvChain[2].b_dly_even_reg[*]/D"]


#Weights chain3
set_multicycle_path -setup 2 -end \
-from [get_pins "m_conv_top/inst_pea/GenOcp[*].u_pe/GenMultCvb.GenMultech[*].u_conv_block/GenConvChain[3].inst_dly_wgt_0/d_r_reg[1][*]*/C"] \
-to   [get_pins "m_conv_top/inst_pea/GenOcp[*].u_pe/GenMultCvb.GenMultech[*].u_conv_block/GenConvChain[3].b_dly_even_reg[*]/D"]
set_multicycle_path -hold 1 -end \
-from [get_pins "m_conv_top/inst_pea/GenOcp[*].u_pe/GenMultCvb.GenMultech[*].u_conv_block/GenConvChain[3].inst_dly_wgt_0/d_r_reg[1][*]*/C"] \
-to   [get_pins "m_conv_top/inst_pea/GenOcp[*].u_pe/GenMultCvb.GenMultech[*].u_conv_block/GenConvChain[3].b_dly_even_reg[*]/D"]
set_multicycle_path -setup 2 -end \
-from [get_pins "m_conv_top/inst_pea/GenOcp[*].u_pe/GenMultCvb.GenMultech[*].u_conv_block/GenConvChain[3].inst_dly_wgt_0/d_r_reg[1][*]*/C"] \
-to   [get_pins "m_conv_top/inst_pea/GenOcp[*].u_pe/GenMultCvb.GenMultech[*].u_conv_block/GenConvChain[3].b_dly_even_reg[*]/D"]
set_multicycle_path -hold 1 -end \
-from [get_pins "m_conv_top/inst_pea/GenOcp[*].u_pe/GenMultCvb.GenMultech[*].u_conv_block/GenConvChain[3].inst_dly_wgt_0/d_r_reg[1][*]*/C"] \
-to   [get_pins "m_conv_top/inst_pea/GenOcp[*].u_pe/GenMultCvb.GenMultech[*].u_conv_block/GenConvChain[3].b_dly_even_reg[*]/D"]

set_multicycle_path -setup 2 -end \
-from [get_pins "m_conv_top/inst_pea/GenOcp[*].u_pe/GenMultCvb.GenMultech[*].u_conv_block/GenConvChain[3].inst_dly_wgt_1/BIT[1].DX.d_r_reg[1][*]*/C"] \
-to   [get_pins "m_conv_top/inst_pea/GenOcp[*].u_pe/GenMultCvb.GenMultech[*].u_conv_block/GenConvChain[3].b_dly_even_reg[*]/D"]
set_multicycle_path -hold 1 -end \
-from [get_pins "m_conv_top/inst_pea/GenOcp[*].u_pe/GenMultCvb.GenMultech[*].u_conv_block/GenConvChain[3].inst_dly_wgt_1/BIT[1].DX.d_r_reg[1][*]*/C"] \
-to   [get_pins "m_conv_top/inst_pea/GenOcp[*].u_pe/GenMultCvb.GenMultech[*].u_conv_block/GenConvChain[3].b_dly_even_reg[*]/D"]
set_multicycle_path -setup 2 -end \
-from [get_pins "m_conv_top/inst_pea/GenOcp[*].u_pe/GenMultCvb.GenMultech[*].u_conv_block/GenConvChain[3].inst_dly_wgt_1/BIT[1].DX.d_r_reg[1][*]*/C"] \
-to   [get_pins "m_conv_top/inst_pea/GenOcp[*].u_pe/GenMultCvb.GenMultech[*].u_conv_block/GenConvChain[3].b_dly_even_reg[*]/D"]
set_multicycle_path -hold 1 -end \
-from [get_pins "m_conv_top/inst_pea/GenOcp[*].u_pe/GenMultCvb.GenMultech[*].u_conv_block/GenConvChain[3].inst_dly_wgt_1/BIT[1].DX.d_r_reg[1][*]*/C"] \
-to   [get_pins "m_conv_top/inst_pea/GenOcp[*].u_pe/GenMultCvb.GenMultech[*].u_conv_block/GenConvChain[3].b_dly_even_reg[*]/D"]


#Weights chain4
set_multicycle_path -setup 2 -end \
-from [get_pins "m_conv_top/inst_pea/GenOcp[*].u_pe/GenMultCvb.GenMultech[*].u_conv_block/GenConvChain[4].inst_dly_wgt_0/d_r_reg[2][*]*/C"] \
-to   [get_pins "m_conv_top/inst_pea/GenOcp[*].u_pe/GenMultCvb.GenMultech[*].u_conv_block/GenConvChain[4].b_dly_even_reg[*]/D"]
set_multicycle_path -hold 1 -end \
-from [get_pins "m_conv_top/inst_pea/GenOcp[*].u_pe/GenMultCvb.GenMultech[*].u_conv_block/GenConvChain[4].inst_dly_wgt_0/d_r_reg[2][*]*/C"] \
-to   [get_pins "m_conv_top/inst_pea/GenOcp[*].u_pe/GenMultCvb.GenMultech[*].u_conv_block/GenConvChain[4].b_dly_even_reg[*]/D"]
set_multicycle_path -setup 2 -end \
-from [get_pins "m_conv_top/inst_pea/GenOcp[*].u_pe/GenMultCvb.GenMultech[*].u_conv_block/GenConvChain[4].inst_dly_wgt_0/d_r_reg[2][*]*/C"] \
-to   [get_pins "m_conv_top/inst_pea/GenOcp[*].u_pe/GenMultCvb.GenMultech[*].u_conv_block/GenConvChain[4].b_dly_even_reg[*]/D"]
set_multicycle_path -hold 1 -end \
-from [get_pins "m_conv_top/inst_pea/GenOcp[*].u_pe/GenMultCvb.GenMultech[*].u_conv_block/GenConvChain[4].inst_dly_wgt_0/d_r_reg[2][*]*/C"] \
-to   [get_pins "m_conv_top/inst_pea/GenOcp[*].u_pe/GenMultCvb.GenMultech[*].u_conv_block/GenConvChain[4].b_dly_even_reg[*]/D"]

set_multicycle_path -setup 2 -end \
-from [get_pins "m_conv_top/inst_pea/GenOcp[*].u_pe/GenMultCvb.GenMultech[*].u_conv_block/GenConvChain[4].inst_dly_wgt_1/BIT[2].DX.d_r_reg[2][*]*/C"] \
-to   [get_pins "m_conv_top/inst_pea/GenOcp[*].u_pe/GenMultCvb.GenMultech[*].u_conv_block/GenConvChain[4].b_dly_even_reg[*]/D"]
set_multicycle_path -hold 1 -end \
-from [get_pins "m_conv_top/inst_pea/GenOcp[*].u_pe/GenMultCvb.GenMultech[*].u_conv_block/GenConvChain[4].inst_dly_wgt_1/BIT[2].DX.d_r_reg[2][*]*/C"] \
-to   [get_pins "m_conv_top/inst_pea/GenOcp[*].u_pe/GenMultCvb.GenMultech[*].u_conv_block/GenConvChain[4].b_dly_even_reg[*]/D"]
set_multicycle_path -setup 2 -end \
-from [get_pins "m_conv_top/inst_pea/GenOcp[*].u_pe/GenMultCvb.GenMultech[*].u_conv_block/GenConvChain[4].inst_dly_wgt_1/BIT[2].DX.d_r_reg[2][*]*/C"] \
-to   [get_pins "m_conv_top/inst_pea/GenOcp[*].u_pe/GenMultCvb.GenMultech[*].u_conv_block/GenConvChain[4].b_dly_even_reg[*]/D"]
set_multicycle_path -hold 1 -end \
-from [get_pins "m_conv_top/inst_pea/GenOcp[*].u_pe/GenMultCvb.GenMultech[*].u_conv_block/GenConvChain[4].inst_dly_wgt_1/BIT[2].DX.d_r_reg[2][*]*/C"] \
-to   [get_pins "m_conv_top/inst_pea/GenOcp[*].u_pe/GenMultCvb.GenMultech[*].u_conv_block/GenConvChain[4].b_dly_even_reg[*]/D"]

#Weights chain5
set_multicycle_path -setup 2 -end \
-from [get_pins "m_conv_top/inst_pea/GenOcp[*].u_pe/GenMultCvb.GenMultech[*].u_conv_block/GenConvChain[5].inst_dly_wgt_0/d_r_reg[2][*]*/C"] \
-to   [get_pins "m_conv_top/inst_pea/GenOcp[*].u_pe/GenMultCvb.GenMultech[*].u_conv_block/GenConvChain[5].b_dly_even_reg[*]/D"]
set_multicycle_path -hold 1 -end \
-from [get_pins "m_conv_top/inst_pea/GenOcp[*].u_pe/GenMultCvb.GenMultech[*].u_conv_block/GenConvChain[5].inst_dly_wgt_0/d_r_reg[2][*]*/C"] \
-to   [get_pins "m_conv_top/inst_pea/GenOcp[*].u_pe/GenMultCvb.GenMultech[*].u_conv_block/GenConvChain[5].b_dly_even_reg[*]/D"]
set_multicycle_path -setup 2 -end \
-from [get_pins "m_conv_top/inst_pea/GenOcp[*].u_pe/GenMultCvb.GenMultech[*].u_conv_block/GenConvChain[5].inst_dly_wgt_0/d_r_reg[2][*]*/C"] \
-to   [get_pins "m_conv_top/inst_pea/GenOcp[*].u_pe/GenMultCvb.GenMultech[*].u_conv_block/GenConvChain[5].b_dly_even_reg[*]/D"]
set_multicycle_path -hold 1 -end \
-from [get_pins "m_conv_top/inst_pea/GenOcp[*].u_pe/GenMultCvb.GenMultech[*].u_conv_block/GenConvChain[5].inst_dly_wgt_0/d_r_reg[2][*]*/C"] \
-to   [get_pins "m_conv_top/inst_pea/GenOcp[*].u_pe/GenMultCvb.GenMultech[*].u_conv_block/GenConvChain[5].b_dly_even_reg[*]/D"]

set_multicycle_path -setup 2 -end \
-from [get_pins "m_conv_top/inst_pea/GenOcp[*].u_pe/GenMultCvb.GenMultech[*].u_conv_block/GenConvChain[5].inst_dly_wgt_1/BIT[2].DX.d_r_reg[2][*]*/C"] \
-to   [get_pins "m_conv_top/inst_pea/GenOcp[*].u_pe/GenMultCvb.GenMultech[*].u_conv_block/GenConvChain[5].b_dly_even_reg[*]/D"]
set_multicycle_path -hold 1 -end \
-from [get_pins "m_conv_top/inst_pea/GenOcp[*].u_pe/GenMultCvb.GenMultech[*].u_conv_block/GenConvChain[5].inst_dly_wgt_1/BIT[2].DX.d_r_reg[2][*]*/C"] \
-to   [get_pins "m_conv_top/inst_pea/GenOcp[*].u_pe/GenMultCvb.GenMultech[*].u_conv_block/GenConvChain[5].b_dly_even_reg[*]/D"]
set_multicycle_path -setup 2 -end \
-from [get_pins "m_conv_top/inst_pea/GenOcp[*].u_pe/GenMultCvb.GenMultech[*].u_conv_block/GenConvChain[5].inst_dly_wgt_1/BIT[2].DX.d_r_reg[2][*]*/C"] \
-to   [get_pins "m_conv_top/inst_pea/GenOcp[*].u_pe/GenMultCvb.GenMultech[*].u_conv_block/GenConvChain[5].b_dly_even_reg[*]/D"]
set_multicycle_path -hold 1 -end \
-from [get_pins "m_conv_top/inst_pea/GenOcp[*].u_pe/GenMultCvb.GenMultech[*].u_conv_block/GenConvChain[5].inst_dly_wgt_1/BIT[2].DX.d_r_reg[2][*]*/C"] \
-to   [get_pins "m_conv_top/inst_pea/GenOcp[*].u_pe/GenMultCvb.GenMultech[*].u_conv_block/GenConvChain[5].b_dly_even_reg[*]/D"]

#Weights chain6
set_multicycle_path -setup 2 -end \
-from [get_pins "m_conv_top/inst_pea/GenOcp[*].u_pe/GenMultCvb.GenMultech[*].u_conv_block/GenConvChain[6].inst_dly_wgt_0/d_r_reg[3][*]*/C"] \
-to   [get_pins "m_conv_top/inst_pea/GenOcp[*].u_pe/GenMultCvb.GenMultech[*].u_conv_block/GenConvChain[6].b_dly_even_reg[*]/D"]
set_multicycle_path -hold 1 -end \
-from [get_pins "m_conv_top/inst_pea/GenOcp[*].u_pe/GenMultCvb.GenMultech[*].u_conv_block/GenConvChain[6].inst_dly_wgt_0/d_r_reg[3][*]*/C"] \
-to   [get_pins "m_conv_top/inst_pea/GenOcp[*].u_pe/GenMultCvb.GenMultech[*].u_conv_block/GenConvChain[6].b_dly_even_reg[*]/D"]
set_multicycle_path -setup 2 -end \
-from [get_pins "m_conv_top/inst_pea/GenOcp[*].u_pe/GenMultCvb.GenMultech[*].u_conv_block/GenConvChain[6].inst_dly_wgt_0/d_r_reg[3][*]*/C"] \
-to   [get_pins "m_conv_top/inst_pea/GenOcp[*].u_pe/GenMultCvb.GenMultech[*].u_conv_block/GenConvChain[6].b_dly_even_reg[*]/D"]
set_multicycle_path -hold 1 -end \
-from [get_pins "m_conv_top/inst_pea/GenOcp[*].u_pe/GenMultCvb.GenMultech[*].u_conv_block/GenConvChain[6].inst_dly_wgt_0/d_r_reg[3][*]*/C"] \
-to   [get_pins "m_conv_top/inst_pea/GenOcp[*].u_pe/GenMultCvb.GenMultech[*].u_conv_block/GenConvChain[6].b_dly_even_reg[*]/D"]

set_multicycle_path -setup 2 -end \
-from [get_pins "m_conv_top/inst_pea/GenOcp[*].u_pe/GenMultCvb.GenMultech[*].u_conv_block/GenConvChain[6].inst_dly_wgt_1/BIT[3].DX.d_r_reg[3][*]*/C"] \
-to   [get_pins "m_conv_top/inst_pea/GenOcp[*].u_pe/GenMultCvb.GenMultech[*].u_conv_block/GenConvChain[6].b_dly_even_reg[*]/D"]
set_multicycle_path -hold 1 -end \
-from [get_pins "m_conv_top/inst_pea/GenOcp[*].u_pe/GenMultCvb.GenMultech[*].u_conv_block/GenConvChain[6].inst_dly_wgt_1/BIT[3].DX.d_r_reg[3][*]*/C"] \
-to   [get_pins "m_conv_top/inst_pea/GenOcp[*].u_pe/GenMultCvb.GenMultech[*].u_conv_block/GenConvChain[6].b_dly_even_reg[*]/D"]
set_multicycle_path -setup 2 -end \
-from [get_pins "m_conv_top/inst_pea/GenOcp[*].u_pe/GenMultCvb.GenMultech[*].u_conv_block/GenConvChain[6].inst_dly_wgt_1/BIT[3].DX.d_r_reg[3][*]*/C"] \
-to   [get_pins "m_conv_top/inst_pea/GenOcp[*].u_pe/GenMultCvb.GenMultech[*].u_conv_block/GenConvChain[6].b_dly_even_reg[*]/D"]
set_multicycle_path -hold 1 -end \
-from [get_pins "m_conv_top/inst_pea/GenOcp[*].u_pe/GenMultCvb.GenMultech[*].u_conv_block/GenConvChain[6].inst_dly_wgt_1/BIT[3].DX.d_r_reg[3][*]*/C"] \
-to   [get_pins "m_conv_top/inst_pea/GenOcp[*].u_pe/GenMultCvb.GenMultech[*].u_conv_block/GenConvChain[6].b_dly_even_reg[*]/D"]

#Weights chain7
set_multicycle_path -setup 2 -end \
-from [get_pins "m_conv_top/inst_pea/GenOcp[*].u_pe/GenMultCvb.GenMultech[*].u_conv_block/GenConvChain[7].inst_dly_wgt_0/d_r_reg[3][*]*/C"] \
-to   [get_pins "m_conv_top/inst_pea/GenOcp[*].u_pe/GenMultCvb.GenMultech[*].u_conv_block/GenConvChain[7].b_dly_even_reg[*]/D"]
set_multicycle_path -hold 1 -end \
-from [get_pins "m_conv_top/inst_pea/GenOcp[*].u_pe/GenMultCvb.GenMultech[*].u_conv_block/GenConvChain[7].inst_dly_wgt_0/d_r_reg[3][*]*/C"] \
-to   [get_pins "m_conv_top/inst_pea/GenOcp[*].u_pe/GenMultCvb.GenMultech[*].u_conv_block/GenConvChain[7].b_dly_even_reg[*]/D"]
set_multicycle_path -setup 2 -end \
-from [get_pins "m_conv_top/inst_pea/GenOcp[*].u_pe/GenMultCvb.GenMultech[*].u_conv_block/GenConvChain[7].inst_dly_wgt_0/d_r_reg[3][*]*/C"] \
-to   [get_pins "m_conv_top/inst_pea/GenOcp[*].u_pe/GenMultCvb.GenMultech[*].u_conv_block/GenConvChain[7].b_dly_even_reg[*]/D"]
set_multicycle_path -hold 1 -end \
-from [get_pins "m_conv_top/inst_pea/GenOcp[*].u_pe/GenMultCvb.GenMultech[*].u_conv_block/GenConvChain[7].inst_dly_wgt_0/d_r_reg[3][*]*/C"] \
-to   [get_pins "m_conv_top/inst_pea/GenOcp[*].u_pe/GenMultCvb.GenMultech[*].u_conv_block/GenConvChain[7].b_dly_even_reg[*]/D"]

set_multicycle_path -setup 2 -end \
-from [get_pins "m_conv_top/inst_pea/GenOcp[*].u_pe/GenMultCvb.GenMultech[*].u_conv_block/GenConvChain[7].inst_dly_wgt_1/BIT[3].DX.d_r_reg[3][*]*/C"] \
-to   [get_pins "m_conv_top/inst_pea/GenOcp[*].u_pe/GenMultCvb.GenMultech[*].u_conv_block/GenConvChain[7].b_dly_even_reg[*]/D"]
set_multicycle_path -hold 1 -end \
-from [get_pins "m_conv_top/inst_pea/GenOcp[*].u_pe/GenMultCvb.GenMultech[*].u_conv_block/GenConvChain[7].inst_dly_wgt_1/BIT[3].DX.d_r_reg[3][*]*/C"] \
-to   [get_pins "m_conv_top/inst_pea/GenOcp[*].u_pe/GenMultCvb.GenMultech[*].u_conv_block/GenConvChain[7].b_dly_even_reg[*]/D"]
set_multicycle_path -setup 2 -end \
-from [get_pins "m_conv_top/inst_pea/GenOcp[*].u_pe/GenMultCvb.GenMultech[*].u_conv_block/GenConvChain[7].inst_dly_wgt_1/BIT[3].DX.d_r_reg[3][*]*/C"] \
-to   [get_pins "m_conv_top/inst_pea/GenOcp[*].u_pe/GenMultCvb.GenMultech[*].u_conv_block/GenConvChain[7].b_dly_even_reg[*]/D"]
set_multicycle_path -hold 1 -end \
-from [get_pins "m_conv_top/inst_pea/GenOcp[*].u_pe/GenMultCvb.GenMultech[*].u_conv_block/GenConvChain[7].inst_dly_wgt_1/BIT[3].DX.d_r_reg[3][*]*/C"] \
-to   [get_pins "m_conv_top/inst_pea/GenOcp[*].u_pe/GenMultCvb.GenMultech[*].u_conv_block/GenConvChain[7].b_dly_even_reg[*]/D"]

#20190105, add bias's path's MCP
set_multicycle_path -setup 4 -from [get_pins "m_conv_top/inst_reader/inst_rd_bias/bias_r_reg[*]/C"]      -to  [get_pins "m_conv_top/inst_pea/GenOcp[*].pe_bias0_reg[*]/D"]
set_multicycle_path -hold  3 -from [get_pins "m_conv_top/inst_reader/inst_rd_bias/bias_r_reg[*]/C"]      -to  [get_pins "m_conv_top/inst_pea/GenOcp[*].pe_bias0_reg[*]/D"]
set_multicycle_path -setup 4 -from [get_pins "m_conv_top/inst_ins_parser/pipe_reg_bias_shf_r_reg[*]/C"]  -to  [get_pins "m_conv_top/inst_pea/GenOcp[*].pe_bias0_reg[*]/D"]
set_multicycle_path -hold  3 -from [get_pins "m_conv_top/inst_ins_parser/pipe_reg_bias_shf_r_reg[*]/C"]  -to  [get_pins "m_conv_top/inst_pea/GenOcp[*].pe_bias0_reg[*]/D"]

set_multicycle_path -setup 4 -from [get_pins "m_conv_top/inst_reader/inst_rd_bias/bias_r_reg[*]/C"]      -to  [get_pins "m_conv_top/inst_pea/GenOcp[*].pe_bias1_reg[*]/D"]
set_multicycle_path -hold  3 -from [get_pins "m_conv_top/inst_reader/inst_rd_bias/bias_r_reg[*]/C"]      -to  [get_pins "m_conv_top/inst_pea/GenOcp[*].pe_bias1_reg[*]/D"]
set_multicycle_path -setup 4 -from [get_pins "m_conv_top/inst_ins_parser/pipe_reg_bias_shf_r_reg[*]/C"]  -to  [get_pins "m_conv_top/inst_pea/GenOcp[*].pe_bias1_reg[*]/D"]
set_multicycle_path -hold  3 -from [get_pins "m_conv_top/inst_ins_parser/pipe_reg_bias_shf_r_reg[*]/C"]  -to  [get_pins "m_conv_top/inst_pea/GenOcp[*].pe_bias1_reg[*]/D"]


#2019-02-27, add MCP for channel auth mode selection
#FIXME: need SunChong to confirm
#set_multicycle_path -setup 2 -from [get_pins "m_wgt_loader/loader/bank_writer/is_chs_aug_mode_reg/C"]
#set_multicycle_path -hold  1 -from [get_pins "m_wgt_loader/loader/bank_writer/is_chs_aug_mode_reg/C"]
#
#set_multicycle_path -setup 2 -from [get_pins "m_fm_loader/loader*/bank_writer/is_chs_aug_mode_reg/C"]
#set_multicycle_path -hold  1 -from [get_pins "m_fm_loader/loader*/bank_writer/is_chs_aug_mode_reg/C"]

#2019-02-27, add MCP for CFG selection in loader
#FIXME: need SunChong to confirm
#set_multicycle_path -setup 2 -from [get_pins " \
#m_wgt_loader/loader/instr_dispatch/cfg_buf_wr_bytes_num_reg[*]/C \
#m_wgt_loader/loader/instr_dispatch/cfg_jump_wr_reg[*]/C \
#m_wgt_loader/loader/instr_dispatch/cfg_jump_rd_reg[*]/C \
#m_wgt_loader/loader/instr_dispatch/cfg_length_reg[*]/C \
#m_wgt_loader/loader/instr_dispatch/cfg_in_chs_reg[*]/C  \
#m_wgt_loader/loader/instr_dispatch/cfg_pad_start_reg[*]/C \
#m_wgt_loader/loader/instr_dispatch/cfg_buf_avg_mode_reg[*]/C \
#m_wgt_loader/loader/instr_dispatch/cfg_buf_wr_bank_id_reg[*]/C"]
#set_multicycle_path -hold  1 -from [get_pins " \
#m_wgt_loader/loader/instr_dispatch/cfg_buf_wr_bytes_num_reg[*]/C \
#m_wgt_loader/loader/instr_dispatch/cfg_jump_wr_reg[*]/C \
#m_wgt_loader/loader/instr_dispatch/cfg_jump_rd_reg[*]/C \
#m_wgt_loader/loader/instr_dispatch/cfg_length_reg[*]/C \
#m_wgt_loader/loader/instr_dispatch/cfg_in_chs_reg[*]/C  \
#m_wgt_loader/loader/instr_dispatch/cfg_pad_start_reg[*]/C \
#m_wgt_loader/loader/instr_dispatch/cfg_buf_avg_mode_reg[*]/C \
#m_wgt_loader/loader/instr_dispatch/cfg_buf_wr_bank_id_reg[*]/C"]
#
#set_multicycle_path -setup 2 -from [get_pins " \
#m_fm_loader/loader*/instr_dispatch/cfg_buf_wr_bytes_num_reg[*]/C \
#m_fm_loader/loader*/instr_dispatch/cfg_jump_wr_reg[*]/C \
#m_fm_loader/loader*/instr_dispatch/cfg_jump_rd_reg[*]/C \
#m_fm_loader/loader*/instr_dispatch/cfg_length_reg[*]/C \
#m_fm_loader/loader*/instr_dispatch/cfg_in_chs_reg[*]/C \
#m_fm_loader/loader*/instr_dispatch/cfg_pad_idx_reg[*]/C \
#m_fm_loader/loader*/instr_dispatch/cfg_pad_start_reg[*]/C \
#m_fm_loader/loader*/instr_dispatch/cfg_pad_end_reg[*]/C \
#m_fm_loader/loader*/instr_dispatch/cfg_buf_avg_mode_reg[*]/C \
#m_fm_loader/loader*/instr_dispatch/cfg_buf_wr_bank_id_reg[*]/C"]
#set_multicycle_path -hold  1 -from [get_pins " \
#m_fm_loader/loader*/instr_dispatch/cfg_buf_wr_bytes_num_reg[*]/C \
#m_fm_loader/loader*/instr_dispatch/cfg_jump_wr_reg[*]/C \
#m_fm_loader/loader*/instr_dispatch/cfg_jump_rd_reg[*]/C \
#m_fm_loader/loader*/instr_dispatch/cfg_length_reg[*]/C \
#m_fm_loader/loader*/instr_dispatch/cfg_in_chs_reg[*]/C \
#m_fm_loader/loader*/instr_dispatch/cfg_pad_idx_reg[*]/C \
#m_fm_loader/loader*/instr_dispatch/cfg_pad_start_reg[*]/C \
#m_fm_loader/loader*/instr_dispatch/cfg_pad_end_reg[*]/C \
#m_fm_loader/loader*/instr_dispatch/cfg_buf_avg_mode_reg[*]/C \
#m_fm_loader/loader*/instr_dispatch/cfg_buf_wr_bank_id_reg[*]/C"]

##2019-02-28, add MCP for ELE's initial instruction 
##FIXME: need Enshan to confirm
#set_multicycle_path -setup 2 -from [get_pins "m_misc_top/ele_top_wrapper/ele_instr_top/reg_jump_read_m1_reg[*][*]/C"]
#set_multicycle_path -hold  1 -from [get_pins "m_misc_top/ele_top_wrapper/ele_instr_top/reg_jump_read_m1_reg[*][*]/C"]
##set_multicycle_path -setup 2 -from [get_pins "m_misc_top/ele_top_wrapper/ele_instr_top/reg_jump_read_endl_m1_reg[*][*]/C"]
##set_multicycle_path -hold  1 -from [get_pins "m_misc_top/ele_top_wrapper/ele_instr_top/reg_jump_read_endl_m1_reg[*][*]/C"]
#
#set_multicycle_path -setup 2 -from [get_pins "m_misc_top/ele_top_wrapper/ele_instr_top/reg_shift_read_reg[*][*]/C"]
#set_multicycle_path -hold  1 -from [get_pins "m_misc_top/ele_top_wrapper/ele_instr_top/reg_shift_read_reg[*][*]/C"]
#
#set_multicycle_path -setup 2 -from [get_pins "m_misc_top/ele_top_wrapper/ele_instr_top/reg_bank_id_in_reg[*][*]/C"]
#set_multicycle_path -hold  1 -from [get_pins "m_misc_top/ele_top_wrapper/ele_instr_top/reg_bank_id_in_reg[*][*]/C"]
#
#set_multicycle_path -setup 2 -from [get_pins "m_misc_top/ele_top_wrapper/ele_instr_top/reg_bank_addr_in_reg[*][*]/C"]
#set_multicycle_path -hold  1 -from [get_pins "m_misc_top/ele_top_wrapper/ele_instr_top/reg_bank_addr_in_reg[*][*]/C"]
#
##set_multicycle_path -setup 2 -from [get_pins "m_misc_top/ele_top_wrapper/ele_instr_top/reg_jump_write_m1_reg[*][*]/C"]
##set_multicycle_path -hold  1 -from [get_pins "m_misc_top/ele_top_wrapper/ele_instr_top/reg_jump_write_m1_reg[*][*]/C"]
##set_multicycle_path -setup 2 -from [get_pins "m_misc_top/ele_top_wrapper/ele_instr_top/reg_jump_write_endl_m1_reg[*][*]/C"]
##set_multicycle_path -hold  1 -from [get_pins "m_misc_top/ele_top_wrapper/ele_instr_top/reg_jump_write_endl_m1_reg[*][*]/C"]
#
##set_multicycle_path -setup 2 -from [get_pins "m_misc_top/ele_top_wrapper/ele_instr_top/reg_shift_write_reg[*][*]/C"]
##set_multicycle_path -hold  1 -from [get_pins "m_misc_top/ele_top_wrapper/ele_instr_top/reg_shift_write_reg[*][*]/C"]

#2019-03-09, add MCP for DWC/Conv's relu shifting
set_multicycle_path -setup 2 -from [get_pins "m_conv_top/inst_pea/GenNLP[*].u_nl/shf_cut_dly_reg[*]/C"] -to [get_pins "m_conv_top/inst_pea/GenNLP[*].u_nl/relu_res_reg[*]/D"]
set_multicycle_path -hold  1 -from [get_pins "m_conv_top/inst_pea/GenNLP[*].u_nl/shf_cut_dly_reg[*]/C"] -to [get_pins "m_conv_top/inst_pea/GenNLP[*].u_nl/relu_res_reg[*]/D"]

#2019-3-22: Robin confirmed
set_multicycle_path -setup 2 -from [get_pins "dwc_block.m_dwc_top/inst_pe_dwc/g_add_cp[*].g_add_pp[*].u_dwc_conv_chain_unit/u_nl_*/shf_cut_dly_reg[*]/C"] -to [get_pins "dwc_block.m_dwc_top/inst_pe_dwc/g_add_cp[*].g_add_pp[*].u_dwc_conv_chain_unit/u_nl_*/relu_res_reg[*]/D"]
set_multicycle_path -hold  1 -from [get_pins "dwc_block.m_dwc_top/inst_pe_dwc/g_add_cp[*].g_add_pp[*].u_dwc_conv_chain_unit/u_nl_*/shf_cut_dly_reg[*]/C"] -to [get_pins "dwc_block.m_dwc_top/inst_pe_dwc/g_add_cp[*].g_add_pp[*].u_dwc_conv_chain_unit/u_nl_*/relu_res_reg[*]/D"]

#FIXME
#2019-03-13, TEMP XDC need designers review
#sed -i 's#^.*reg\s\[DATA_BW\s*-1:0\]\s*pipe_reg_nl_relu6_r#(* keep = "true" *) reg \[DATA_BW            -1:0\] pipe_reg_nl_relu6_r#' ../ip/dpu/rtl/cnn/conv/reader/conv_ins_parser.v
#sed -i 's#^.*reg\s\[NL_TYPE_BW\s*-1:0\]\s*pipe_reg_nl_type_r#(* keep = "true" *) reg \[NL_TYPE_BW         -1:0\] pipe_reg_nl_type_r#' ../ip/dpu/rtl/cnn/conv/reader/conv_ins_parser.v
#sed -i 's#^.*reg\s*\[NL_DEC_BW\s*-1:0\]\s*reg_nl_type_r#(* keep = "true" *) reg \[NL_DEC_BW    -1:0\] reg_nl_type_r #' ../ip/dpu/rtl/cnn/depthwise/dwc_ins_parser.v
#sed -i 's#^.*reg\s*\[DATA_BW\s*-1:0\]\s*reg_nl_relu6_r#(* keep = "true" *) reg \[DATA_BW-1:0\] reg_nl_relu6_r#' ../ip/dpu/rtl/cnn/depthwise/dwc_ins_parser.v
set_multicycle_path -setup 2 -from [get_pins " \
m_conv_top/inst_ins_parser/pipe_reg_nl_relu6_r_reg[*]/C \
m_conv_top/inst_ins_parser/pipe_reg_nl_type_r_reg[*]/C \
"]
set_multicycle_path -hold  1 -from [get_pins " \
m_conv_top/inst_ins_parser/pipe_reg_nl_relu6_r_reg[*]/C \
m_conv_top/inst_ins_parser/pipe_reg_nl_type_r_reg[*]/C \
"]

#2019-3-22: Robin confirmed
set_multicycle_path -setup 2 -from [get_pins " \
dwc_block.m_dwc_top/u_dwc_ins_parser/reg_nl_type_r_reg[*]/C \
dwc_block.m_dwc_top/u_dwc_ins_parser/reg_nl_relu6_r_reg[*]/C \
"]
set_multicycle_path -hold  1 -from [get_pins " \
dwc_block.m_dwc_top/u_dwc_ins_parser/reg_nl_type_r_reg[*]/C \
dwc_block.m_dwc_top/u_dwc_ins_parser/reg_nl_relu6_r_reg[*]/C \
"]

#2019-04-01, add false path for ACLK_DR's last-level reset logic
#set_multicycle_path -setup 2 -from [get_pins " \
#m_conv_top/inst_pea/GenOcp[*].u_pe/rst_n_rr_reg/C \
#"]
#set_multicycle_path -hold  1 -from [get_pins " \
#m_conv_top/inst_pea/GenOcp[*].u_pe/rst_n_rr_reg/C \
#"]

#2019-5-30, add MCP for below path
set_multicycle_path -setup 2 -to [get_pins "\
m_conv_top/inst_reader/inst_rd_img/u_addr_gen/GenAddr[*].area_w_reg[*]/D \
"]
set_multicycle_path -hold 1 -to [get_pins "\
m_conv_top/inst_reader/inst_rd_img/u_addr_gen/GenAddr[*].area_w_reg[*]/D \
"]


set_multicycle_path -setup 2 -thr [get_pins \
"gen_load_module.m_load_top/gen_loader_port[*].gen_loader_aug[*].loader/ddr_reader/total_nbyte_rd_reg/* \
 gen_load_module.m_load_top/gen_loader_port[*].gen_loader_aug[*].loader/ddr_reader/total_nbyte_wr_reg/* "\
-filter direction==in
]
set_multicycle_path -hold  1 -thr [get_pins \
"gen_load_module.m_load_top/gen_loader_port[*].gen_loader_aug[*].loader/ddr_reader/total_nbyte_rd_reg/* \
 gen_load_module.m_load_top/gen_loader_port[*].gen_loader_aug[*].loader/ddr_reader/total_nbyte_wr_reg/* "\
-filter direction==in
]



##2019-07-03, add MCP between OP#Conv's Weights Buf#Conv's Weights Buf_INIT and OP_POOL because scheduler need about 10 clock cycles to confirm dependencies
##in pool_axi_aw.v
#set_multicycle_path -setup 2 -from [get_pins "\
#m_misc_top/pool_top_wrapper/POOL[*].I_EQ_*.pool_top_writing/pool_axi_aw/Q_stride_offset_out_reg[*]/C \
#m_misc_top/pool_top_wrapper/POOL[*].I_EQ_*.pool_top_writing/pool_axi_aw/Q_jump_write_endl_reg[*]/C \
#"] \
#-to  [get_pins "\
#m_misc_top/pool_top_wrapper/POOL[*].I_EQ_*.pool_top_writing/pool_axi_aw/int_bank_addr_reg[*]/D \
#m_misc_top/pool_top_wrapper/POOL[*].I_EQ_*.pool_top_writing/pool_axi_aw/Q_bank_addr_out_reg[*]/D \
#"]
#set_multicycle_path -hold 1 -from [get_pins "\
#m_misc_top/pool_top_wrapper/POOL[*].I_EQ_*.pool_top_writing/pool_axi_aw/Q_stride_offset_out_reg[*]/C \
#m_misc_top/pool_top_wrapper/POOL[*].I_EQ_*.pool_top_writing/pool_axi_aw/Q_jump_write_endl_reg[*]/C \
#"] \
#-to  [get_pins "\
#m_misc_top/pool_top_wrapper/POOL[*].I_EQ_*.pool_top_writing/pool_axi_aw/int_bank_addr_reg[*]/D \
#m_misc_top/pool_top_wrapper/POOL[*].I_EQ_*.pool_top_writing/pool_axi_aw/Q_bank_addr_out_reg[*]/D \
#"]
#set_multicycle_path -setup 2 -from [get_pins "\
#m_misc_top/pool_top_wrapper/POOL[*].I_EQ_*.pool_top_writing/pool_axi_aw/Q_stride_out_reg[*]/C \
#m_misc_top/pool_top_wrapper/POOL[*].I_EQ_*.pool_top_writing/pool_axi_aw/Q_jump_write_endl_reg[*]/C \
#"] \
#-to  [get_pins "\
#m_misc_top/pool_top_wrapper/POOL[*].I_EQ_*.pool_top_writing/pool_axi_aw/int_bias_jump[*]/D \
#"]
#set_multicycle_path -hold 1 -from [get_pins "\
#m_misc_top/pool_top_wrapper/POOL[*].I_EQ_*.pool_top_writing/pool_axi_aw/Q_stride_out_reg[*]/C \
#m_misc_top/pool_top_wrapper/POOL[*].I_EQ_*.pool_top_writing/pool_axi_aw/Q_jump_write_endl_reg[*]/C \
#"] \
#-to  [get_pins "\
#m_misc_top/pool_top_wrapper/POOL[*].I_EQ_*.pool_top_writing/pool_axi_aw/int_bias_jump[*]/D \
#"]
##in pool_axi_ar.v
#set_multicycle_path -setup 2 -from [get_pins "\
#m_misc_top/pool_top_wrapper/POOL[*].I_EQ_*.pool_top_reading/pool_axi_ar/Q_stride_w_reg[*]/C \
#m_misc_top/pool_top_wrapper/POOL[*].I_EQ_*.pool_top_reading/pool_axi_ar/Q_jump_read_reg[*]/C \
#"] \
#-to [get_pins "\
#m_misc_top/pool_top_wrapper/POOL[*].I_EQ_*.pool_top_reading/pool_axi_ar/int_dot_jump_reg[*]/D \
#"]
#set_multicycle_path -hold 1 -from [get_pins "\
#m_misc_top/pool_top_wrapper/POOL[*].I_EQ_*.pool_top_reading/pool_axi_ar/Q_stride_w_reg[*]/C \
#m_misc_top/pool_top_wrapper/POOL[*].I_EQ_*.pool_top_reading/pool_axi_ar/Q_jump_read_reg[*]/C \
#"] \
#-to [get_pins "\
#m_misc_top/pool_top_wrapper/POOL[*].I_EQ_*.pool_top_reading/pool_axi_ar/int_dot_jump_reg[*]/D \
#"]
#set_multicycle_path -setup 2 -from [get_pins "\
#m_misc_top/pool_top_wrapper/POOL[*].I_EQ_*.pool_top_reading/pool_axi_ar/Q_stride_h_reg[*]/C \
#m_misc_top/pool_top_wrapper/POOL[*].I_EQ_*.pool_top_reading/pool_axi_ar/Q_jump_read_endl_reg[*]/C \
#"] \
#-to [get_pins "\
#m_misc_top/pool_top_wrapper/POOL[*].I_EQ_*.pool_top_reading/pool_axi_ar/int_bias_jump_reg[*]/D \
#"]
#set_multicycle_path -hold 1 -from [get_pins "\
#m_misc_top/pool_top_wrapper/POOL[*].I_EQ_*.pool_top_reading/pool_axi_ar/Q_stride_h_reg[*]/C \
#m_misc_top/pool_top_wrapper/POOL[*].I_EQ_*.pool_top_reading/pool_axi_ar/Q_jump_read_endl_reg[*]/C \
#"] \
#-to [get_pins "\
#m_misc_top/pool_top_wrapper/POOL[*].I_EQ_*.pool_top_reading/pool_axi_ar/int_bias_jump_reg[*]/D \
#"]
##in pool_axi_r.v
#set_multicycle_path -setup 2 -from [get_pins "\
#m_misc_top/ele_top_wrapper/ele_instr_top/reg_length_m1_reg[*]/C \
#m_misc_top/pool_top_wrapper/POOL[*].I_EQ_*.pool_top_reading/pool_axi_r/Q_stride_w_reg[*]/C \
#m_misc_top/pool_top_wrapper/POOL[*].I_EQ_*.pool_top_reading/pool_axi_r/Q_kernel_w_reg[*]/C \
#"] \
#-to [get_pins "\
#m_misc_top/pool_top_wrapper/POOL[0].I_EQ_0.pool_top_reading/pool_axi_r/r_abs_col_max_reg[*]/D \
#"]
#set_multicycle_path -hold 1 -from [get_pins "\
#m_misc_top/ele_top_wrapper/ele_instr_top/reg_length_m1_reg[*]/C \
#m_misc_top/pool_top_wrapper/POOL[*].I_EQ_*.pool_top_reading/pool_axi_r/Q_stride_w_reg[*]/C \
#m_misc_top/pool_top_wrapper/POOL[*].I_EQ_*.pool_top_reading/pool_axi_r/Q_kernel_w_reg[*]/C \
#"] \
#-to [get_pins "\
#m_misc_top/pool_top_wrapper/POOL[0].I_EQ_0.pool_top_reading/pool_axi_r/r_abs_col_max_reg[*]/D \
#"]
