#20181114, from reg_map's control register
## dpu_resetn
set_false_path -from [get_pins "m_regmap/int_dpu0_scalar03*/C"] -to [get_pins "m_regmap/dpu0_scalar03_Q*/D"] 
## dpu_start
set_false_path -from [get_pins "m_regmap/int_dpu0_scalar02*/C"]
## dpu_interrupt_raw_sts
set_false_path -to [get_pins "m_regmap/int_dpu0_status00_Q0*/D"]

##Reg Maps: read-
set_false_path -from [get_pins "m_regmap/int_dpu0_scalar00*/C"]
set_false_path -from [get_pins "m_regmap/int_dpu0_scalar01*/C"]
set_false_path -from [get_pins "m_regmap/int_dpu0_scalar03*/C"]
set_false_path -from [get_pins "m_regmap/int_dpu0_scalar04*/C"]
set_false_path -from [get_pins "m_regmap/int_dpu0_scalar05*/C"]
set_false_path -from [get_pins "m_regmap/int_dpu0_scalar06*/C"]
set_false_path -from [get_pins "m_regmap/int_dpu0_scalar07*/C"]
set_false_path -from [get_pins "m_regmap/int_dpu0_scalar08*/C"]
set_false_path -from [get_pins "m_regmap/int_dpu0_scalar09*/C"]
set_false_path -from [get_pins "m_regmap/int_dpu0_scalar10*/C"]
set_false_path -from [get_pins "m_regmap/int_dpu0_scalar11*/C"]
set_false_path -from [get_pins "m_regmap/int_dpu0_scalar12*/C"]
set_false_path -from [get_pins "m_regmap/int_dpu0_scalar13*/C"]
set_false_path -from [get_pins "m_regmap/int_dpu0_scalar14*/C"]

set_false_path -from [get_pins "m_regmap/gen_base_addr[*].int_dpu*_axi00_ptr00*/C"]
set_false_path -from [get_pins "m_regmap/gen_base_addr[*].int_dpu*_axi00_ptr01*/C"]
set_false_path -from [get_pins "m_regmap/gen_base_addr[*].int_dpu*_axi00_ptr02*/C"]
set_false_path -from [get_pins "m_regmap/gen_base_addr[*].int_dpu*_axi00_ptr03*/C"]
set_false_path -from [get_pins "m_regmap/gen_base_addr[*].int_dpu*_axi00_ptr04*/C"]
set_false_path -from [get_pins "m_regmap/gen_base_addr[*].int_dpu*_axi00_ptr05*/C"]
set_false_path -from [get_pins "m_regmap/gen_base_addr[*].int_dpu*_axi00_ptr06*/C"]
set_false_path -from [get_pins "m_regmap/gen_base_addr[*].int_dpu*_axi00_ptr07*/C"]
set_false_path -from [get_pins "m_regmap/int_dpu0_axi00_ptr08*/C"]
#set_false_path -from [get_pins "m_regmap/int_dpu0_axi00_ptr09*/C"]
#set_false_path -from [get_pins "m_regmap/int_dpu0_axi00_ptr10*/C"]

##Reg map: read-only
set_false_path -to [get_pins "m_regmap/int_dpu0_status01*/D"]
set_false_path -to [get_pins "m_regmap/int_dpu0_status02*/D"]
set_false_path -to [get_pins "m_regmap/int_dpu0_status03*/D"]
set_false_path -to [get_pins "m_regmap/int_dpu0_status04*/D"]
set_false_path -to [get_pins "m_regmap/int_dpu0_status05*/D"]
set_false_path -to [get_pins "m_regmap/int_dpu0_status06*/D"]
set_false_path -to [get_pins "m_regmap/int_dpu0_status07*/D"]
set_false_path -to [get_pins "m_regmap/int_dpu0_status08*/D"]
#set_false_path -to [get_pins "m_regmap/int_dpu0_status09*/D"]
set_false_path -to [get_pins "m_regmap/int_dpu0_status10*/D"]

#==================================================================================
#From v3me
#2019-03-26, add false path for dpu_interrupt
set_false_path -thr [get_ports -scoped_to_current_instance "dpu_interrupt"]

#2019-05-17, add false path to ap_start
set_false_path -to [get_pins "m_regmap/dpu0_scalar00_Q_reg*/D"]
#2019-05-17, add false path to finish_clr
set_false_path -to [get_pins "m_regmap/dpu0_scalar02_Q_reg*/D"]
#==================================================================================

#2019-06-27, no need here because this constraints has been moved into clock generator
#set_false_path -to [get_pins -hier -filter "name=~ARESETN_DPU_Q1_reg/CLR"]
#set_false_path -to [get_pins -hier -filter "name=~ARESETN_DPU_Q2_reg/CLR"]


#2018-11-09, from Pengbo, add MCP from 666 to 333
set_multicycle_path -setup 2 -start \
-from [get_pins "gen_dpu.m_dpu_*/m_conv_top/inst_pea/GenOcp[*].u_pe/GenMultCvb.GenMultech[*].GenLastCvb.GenLstPeDataGrp[*].add_res_last_tmp_reg_0_reg[*]/C"] \
-to   [get_pins "gen_dpu.m_dpu_*/m_conv_top/inst_pea/GenOcp[*].u_pe/GenMultCvb.GenMultech[*].GenLastCvb.GenLstPeDataGrp[*].add_res_last_tmp_rr_0_reg[*]/D"]
set_multicycle_path -hold 1 -start \
-from [get_pins "gen_dpu.m_dpu_*/m_conv_top/inst_pea/GenOcp[*].u_pe/GenMultCvb.GenMultech[*].GenLastCvb.GenLstPeDataGrp[*].add_res_last_tmp_reg_0_reg[*]/C"] \
-to   [get_pins "gen_dpu.m_dpu_*/m_conv_top/inst_pea/GenOcp[*].u_pe/GenMultCvb.GenMultech[*].GenLastCvb.GenLstPeDataGrp[*].add_res_last_tmp_rr_0_reg[*]/D"]

set_multicycle_path -setup 2 -start \
-from [get_pins "gen_dpu.m_dpu_*/m_conv_top/inst_pea/GenOcp[*].u_pe/GenMultCvb.GenMultech[*].GenLastCvb.GenLstPeDataGrp[*].add_res_last_tmp_reg_1_reg[*]/C"] \
-to   [get_pins "gen_dpu.m_dpu_*/m_conv_top/inst_pea/GenOcp[*].u_pe/GenMultCvb.GenMultech[*].GenLastCvb.GenLstPeDataGrp[*].add_res_last_tmp_rr_1_reg[*]/D"]
set_multicycle_path -hold 1 -start \
-from [get_pins "gen_dpu.m_dpu_*/m_conv_top/inst_pea/GenOcp[*].u_pe/GenMultCvb.GenMultech[*].GenLastCvb.GenLstPeDataGrp[*].add_res_last_tmp_reg_1_reg[*]/C"] \
-to   [get_pins "gen_dpu.m_dpu_*/m_conv_top/inst_pea/GenOcp[*].u_pe/GenMultCvb.GenMultech[*].GenLastCvb.GenLstPeDataGrp[*].add_res_last_tmp_rr_1_reg[*]/D"]


set_multicycle_path -setup 2 -start \
-from [get_pins "gen_dpu.m_dpu_*/m_conv_top/inst_pea/GenOcp[*].u_pe/GenMultCvb.GenMultech[*].GenLastCvb.GenLstPeDataGrp[*].add_res_8_last_reg_0_reg[*]/C"] \
-to   [get_pins "gen_dpu.m_dpu_*/m_conv_top/inst_pea/GenOcp[*].u_pe/GenMultCvb.GenMultech[*].GenLastCvb.GenLstPeDataGrp[*].add_res_8_last_rr_0_reg[*]/D"]
set_multicycle_path -hold 1 -start \
-from [get_pins "gen_dpu.m_dpu_*/m_conv_top/inst_pea/GenOcp[*].u_pe/GenMultCvb.GenMultech[*].GenLastCvb.GenLstPeDataGrp[*].add_res_8_last_reg_0_reg[*]/C"] \
-to   [get_pins "gen_dpu.m_dpu_*/m_conv_top/inst_pea/GenOcp[*].u_pe/GenMultCvb.GenMultech[*].GenLastCvb.GenLstPeDataGrp[*].add_res_8_last_rr_0_reg[*]/D"]

set_multicycle_path -setup 2 -start \
-from [get_pins "gen_dpu.m_dpu_*/m_conv_top/inst_pea/GenOcp[*].u_pe/GenMultCvb.GenMultech[*].GenLastCvb.GenLstPeDataGrp[*].add_res_8_last_reg_1_reg[*]/C"] \
-to   [get_pins "gen_dpu.m_dpu_*/m_conv_top/inst_pea/GenOcp[*].u_pe/GenMultCvb.GenMultech[*].GenLastCvb.GenLstPeDataGrp[*].add_res_8_last_rr_1_reg[*]/D"]
set_multicycle_path -hold 1 -start \
-from [get_pins "gen_dpu.m_dpu_*/m_conv_top/inst_pea/GenOcp[*].u_pe/GenMultCvb.GenMultech[*].GenLastCvb.GenLstPeDataGrp[*].add_res_8_last_reg_1_reg[*]/C"] \
-to   [get_pins "gen_dpu.m_dpu_*/m_conv_top/inst_pea/GenOcp[*].u_pe/GenMultCvb.GenMultech[*].GenLastCvb.GenLstPeDataGrp[*].add_res_8_last_rr_1_reg[*]/D"]

#chain0
set_multicycle_path -setup 2 -end \
-from [get_pins "gen_dpu.m_dpu_*/m_conv_top/inst_pea/GenOcp[*].u_pe/GenMultCvb.GenMultech[0].GenMidCvb.u_conv_block/GenConvChain[0].inst_dly_tdata_01/d_r_reg[0][*]/C"] \
-to   [get_pins "gen_dpu.m_dpu_*/m_conv_top/inst_pea/GenOcp[*].u_pe/GenMultCvb.GenMultech[0].GenMidCvb.u_conv_block/GenConvChain[0].GenConvPair[*].Gen8x8.GenMultDsp.GenFirDsp.u_dsp_unit_fir/DSP48E2_gen.DSP48E2_inst/DSP_PREADD_DATA_INST/DIN[*]"]
set_multicycle_path -hold 1 -end \
-from [get_pins "gen_dpu.m_dpu_*/m_conv_top/inst_pea/GenOcp[*].u_pe/GenMultCvb.GenMultech[0].GenMidCvb.u_conv_block/GenConvChain[0].inst_dly_tdata_01/d_r_reg[0][*]/C"] \
-to   [get_pins "gen_dpu.m_dpu_*/m_conv_top/inst_pea/GenOcp[*].u_pe/GenMultCvb.GenMultech[0].GenMidCvb.u_conv_block/GenConvChain[0].GenConvPair[*].Gen8x8.GenMultDsp.GenFirDsp.u_dsp_unit_fir/DSP48E2_gen.DSP48E2_inst/DSP_PREADD_DATA_INST/DIN[*]"]

set_multicycle_path -setup 2 -end \
-from [get_pins "gen_dpu.m_dpu_*/m_conv_top/inst_pea/GenOcp[*].u_pe/GenMultCvb.GenMultech[0].GenMidCvb.u_conv_block/GenConvChain[0].inst_dly_tdata_01/d_r_reg[0][*]/C"] \
-to   [get_pins "gen_dpu.m_dpu_*/m_conv_top/inst_pea/GenOcp[*].u_pe/GenMultCvb.GenMultech[0].GenMidCvb.u_conv_block/GenConvChain[0].GenConvPair[*].Gen8x8.GenMultDsp.GenFirDsp.u_dsp_unit_fir/DSP48E2_gen.DSP48E2_inst/DSP_A_B_DATA_INST/A[*]"]
set_multicycle_path -hold 1 -end \
-from [get_pins "gen_dpu.m_dpu_*/m_conv_top/inst_pea/GenOcp[*].u_pe/GenMultCvb.GenMultech[0].GenMidCvb.u_conv_block/GenConvChain[0].inst_dly_tdata_01/d_r_reg[0][*]/C"] \
-to   [get_pins "gen_dpu.m_dpu_*/m_conv_top/inst_pea/GenOcp[*].u_pe/GenMultCvb.GenMultech[0].GenMidCvb.u_conv_block/GenConvChain[0].GenConvPair[*].Gen8x8.GenMultDsp.GenFirDsp.u_dsp_unit_fir/DSP48E2_gen.DSP48E2_inst/DSP_A_B_DATA_INST/A[*]"]

set_multicycle_path -setup 2 -end \
-from [get_pins "gen_dpu.m_dpu_*/m_conv_top/inst_pea/GenOcp[*].u_pe/GenMultCvb.GenMultech[1].GenLastCvb.u_conv_block/GenConvChain[0].inst_dly_tdata_01/d_r_reg[0][*]/C"] \
-to   [get_pins "gen_dpu.m_dpu_*/m_conv_top/inst_pea/GenOcp[*].u_pe/GenMultCvb.GenMultech[1].GenLastCvb.u_conv_block/GenConvChain[0].GenConvPair[*].Gen8x8.GenMultDsp.GenFirDsp.u_dsp_unit_fir/DSP48E2_gen.DSP48E2_inst/DSP_PREADD_DATA_INST/DIN[*]"]
set_multicycle_path -hold 1 -end \
-from [get_pins "gen_dpu.m_dpu_*/m_conv_top/inst_pea/GenOcp[*].u_pe/GenMultCvb.GenMultech[1].GenLastCvb.u_conv_block/GenConvChain[0].inst_dly_tdata_01/d_r_reg[0][*]/C"] \
-to   [get_pins "gen_dpu.m_dpu_*/m_conv_top/inst_pea/GenOcp[*].u_pe/GenMultCvb.GenMultech[1].GenLastCvb.u_conv_block/GenConvChain[0].GenConvPair[*].Gen8x8.GenMultDsp.GenFirDsp.u_dsp_unit_fir/DSP48E2_gen.DSP48E2_inst/DSP_PREADD_DATA_INST/DIN[*]"]

set_multicycle_path -setup 2 -end \
-from [get_pins "gen_dpu.m_dpu_*/m_conv_top/inst_pea/GenOcp[*].u_pe/GenMultCvb.GenMultech[1].GenLastCvb.u_conv_block/GenConvChain[0].inst_dly_tdata_01/d_r_reg[0][*]/C"] \
-to   [get_pins "gen_dpu.m_dpu_*/m_conv_top/inst_pea/GenOcp[*].u_pe/GenMultCvb.GenMultech[1].GenLastCvb.u_conv_block/GenConvChain[0].GenConvPair[*].Gen8x8.GenMultDsp.GenFirDsp.u_dsp_unit_fir/DSP48E2_gen.DSP48E2_inst/DSP_A_B_DATA_INST/A[*]"]
set_multicycle_path -hold 1 -end \
-from [get_pins "gen_dpu.m_dpu_*/m_conv_top/inst_pea/GenOcp[*].u_pe/GenMultCvb.GenMultech[1].GenLastCvb.u_conv_block/GenConvChain[0].inst_dly_tdata_01/d_r_reg[0][*]/C"] \
-to   [get_pins "gen_dpu.m_dpu_*/m_conv_top/inst_pea/GenOcp[*].u_pe/GenMultCvb.GenMultech[1].GenLastCvb.u_conv_block/GenConvChain[0].GenConvPair[*].Gen8x8.GenMultDsp.GenFirDsp.u_dsp_unit_fir/DSP48E2_gen.DSP48E2_inst/DSP_A_B_DATA_INST/A[*]"]

#chain 1
set_multicycle_path -setup 2 -end \
-from [get_pins "gen_dpu.m_dpu_*/m_conv_top/inst_pea/GenOcp[*].u_pe/GenMultCvb.GenMultech[0].GenMidCvb.u_conv_block/GenConvChain[1].inst_dly_tdata_01/d_r_reg[0][*]/C"] \
-to   [get_pins "gen_dpu.m_dpu_*/m_conv_top/inst_pea/GenOcp[*].u_pe/GenMultCvb.GenMultech[0].GenMidCvb.u_conv_block/GenConvChain[1].pe_tdata_grp_dly_odd_reg[*]/D"]
set_multicycle_path -hold 1 -end \
-from [get_pins "gen_dpu.m_dpu_*/m_conv_top/inst_pea/GenOcp[*].u_pe/GenMultCvb.GenMultech[0].GenMidCvb.u_conv_block/GenConvChain[1].inst_dly_tdata_01/d_r_reg[0][*]/C"] \
-to   [get_pins "gen_dpu.m_dpu_*/m_conv_top/inst_pea/GenOcp[*].u_pe/GenMultCvb.GenMultech[0].GenMidCvb.u_conv_block/GenConvChain[1].pe_tdata_grp_dly_odd_reg[*]/D"]

set_multicycle_path -setup 2 -end \
-from [get_pins "gen_dpu.m_dpu_*/m_conv_top/inst_pea/GenOcp[*].u_pe/GenMultCvb.GenMultech[1].GenLastCvb.u_conv_block/GenConvChain[1].inst_dly_tdata_01/d_r_reg[0][*]/C"] \
-to   [get_pins "gen_dpu.m_dpu_*/m_conv_top/inst_pea/GenOcp[*].u_pe/GenMultCvb.GenMultech[1].GenLastCvb.u_conv_block/GenConvChain[1].pe_tdata_grp_dly_odd_reg[*]/D"]
set_multicycle_path -hold 1 -end \
-from [get_pins "gen_dpu.m_dpu_*/m_conv_top/inst_pea/GenOcp[*].u_pe/GenMultCvb.GenMultech[1].GenLastCvb.u_conv_block/GenConvChain[1].inst_dly_tdata_01/d_r_reg[0][*]/C"] \
-to   [get_pins "gen_dpu.m_dpu_*/m_conv_top/inst_pea/GenOcp[*].u_pe/GenMultCvb.GenMultech[1].GenLastCvb.u_conv_block/GenConvChain[1].pe_tdata_grp_dly_odd_reg[*]/D"]

#chain2
set_multicycle_path -setup 2 -end \
-from [get_pins "gen_dpu.m_dpu_*/m_conv_top/inst_pea/GenOcp[*].u_pe/GenMultCvb.GenMultech[0].GenMidCvb.u_conv_block/GenConvChain[2].inst_dly_tdata/d_r_reg[1][*]*/C"] \
-to   [get_pins "gen_dpu.m_dpu_*/m_conv_top/inst_pea/GenOcp[*].u_pe/GenMultCvb.GenMultech[0].GenMidCvb.u_conv_block/GenConvChain[2].GenConvPair[*].Gen8x8.GenMultDsp.GenMidDsp.u_dsp_unit_mid/DSP48E2_gen.DSP48E2_inst/DSP_PREADD_DATA_INST/DIN[*]"]
set_multicycle_path -hold 1 -end \
-from [get_pins "gen_dpu.m_dpu_*/m_conv_top/inst_pea/GenOcp[*].u_pe/GenMultCvb.GenMultech[0].GenMidCvb.u_conv_block/GenConvChain[2].inst_dly_tdata/d_r_reg[1][*]*/C"] \
-to   [get_pins "gen_dpu.m_dpu_*/m_conv_top/inst_pea/GenOcp[*].u_pe/GenMultCvb.GenMultech[0].GenMidCvb.u_conv_block/GenConvChain[2].GenConvPair[*].Gen8x8.GenMultDsp.GenMidDsp.u_dsp_unit_mid/DSP48E2_gen.DSP48E2_inst/DSP_PREADD_DATA_INST/DIN[*]"]

set_multicycle_path -setup 2 -end \
-from [get_pins "gen_dpu.m_dpu_*/m_conv_top/inst_pea/GenOcp[*].u_pe/GenMultCvb.GenMultech[0].GenMidCvb.u_conv_block/GenConvChain[2].inst_dly_tdata/d_r_reg[1][*]*/C"] \
-to   [get_pins "gen_dpu.m_dpu_*/m_conv_top/inst_pea/GenOcp[*].u_pe/GenMultCvb.GenMultech[0].GenMidCvb.u_conv_block/GenConvChain[2].GenConvPair[*].Gen8x8.GenMultDsp.GenMidDsp.u_dsp_unit_mid/DSP48E2_gen.DSP48E2_inst/DSP_A_B_DATA_INST/A[*]"]
set_multicycle_path -hold 1 -end \
-from [get_pins "gen_dpu.m_dpu_*/m_conv_top/inst_pea/GenOcp[*].u_pe/GenMultCvb.GenMultech[0].GenMidCvb.u_conv_block/GenConvChain[2].inst_dly_tdata/d_r_reg[1][*]*/C"] \
-to   [get_pins "gen_dpu.m_dpu_*/m_conv_top/inst_pea/GenOcp[*].u_pe/GenMultCvb.GenMultech[0].GenMidCvb.u_conv_block/GenConvChain[2].GenConvPair[*].Gen8x8.GenMultDsp.GenMidDsp.u_dsp_unit_mid/DSP48E2_gen.DSP48E2_inst/DSP_A_B_DATA_INST/A[*]"]

set_multicycle_path -setup 2 -end \
-from [get_pins "gen_dpu.m_dpu_*/m_conv_top/inst_pea/GenOcp[*].u_pe/GenMultCvb.GenMultech[1].GenLastCvb.u_conv_block/GenConvChain[2].inst_dly_tdata/d_r_reg[1][*]*/C"] \
-to   [get_pins "gen_dpu.m_dpu_*/m_conv_top/inst_pea/GenOcp[*].u_pe/GenMultCvb.GenMultech[1].GenLastCvb.u_conv_block/GenConvChain[2].GenConvPair[*].Gen8x8.GenMultDsp.GenMidDsp.u_dsp_unit_mid/DSP48E2_gen.DSP48E2_inst/DSP_PREADD_DATA_INST/DIN[*]"]
set_multicycle_path -hold 1 -end \
-from [get_pins "gen_dpu.m_dpu_*/m_conv_top/inst_pea/GenOcp[*].u_pe/GenMultCvb.GenMultech[1].GenLastCvb.u_conv_block/GenConvChain[2].inst_dly_tdata/d_r_reg[1][*]*/C"] \
-to   [get_pins "gen_dpu.m_dpu_*/m_conv_top/inst_pea/GenOcp[*].u_pe/GenMultCvb.GenMultech[1].GenLastCvb.u_conv_block/GenConvChain[2].GenConvPair[*].Gen8x8.GenMultDsp.GenMidDsp.u_dsp_unit_mid/DSP48E2_gen.DSP48E2_inst/DSP_PREADD_DATA_INST/DIN[*]"]

set_multicycle_path -setup 2 -end \
-from [get_pins "gen_dpu.m_dpu_*/m_conv_top/inst_pea/GenOcp[*].u_pe/GenMultCvb.GenMultech[1].GenLastCvb.u_conv_block/GenConvChain[2].inst_dly_tdata/d_r_reg[1][*]*/C"] \
-to   [get_pins "gen_dpu.m_dpu_*/m_conv_top/inst_pea/GenOcp[*].u_pe/GenMultCvb.GenMultech[1].GenLastCvb.u_conv_block/GenConvChain[2].GenConvPair[*].Gen8x8.GenMultDsp.GenMidDsp.u_dsp_unit_mid/DSP48E2_gen.DSP48E2_inst/DSP_A_B_DATA_INST/A[*]"]
set_multicycle_path -hold 1 -end \
-from [get_pins "gen_dpu.m_dpu_*/m_conv_top/inst_pea/GenOcp[*].u_pe/GenMultCvb.GenMultech[1].GenLastCvb.u_conv_block/GenConvChain[2].inst_dly_tdata/d_r_reg[1][*]*/C"] \
-to   [get_pins "gen_dpu.m_dpu_*/m_conv_top/inst_pea/GenOcp[*].u_pe/GenMultCvb.GenMultech[1].GenLastCvb.u_conv_block/GenConvChain[2].GenConvPair[*].Gen8x8.GenMultDsp.GenMidDsp.u_dsp_unit_mid/DSP48E2_gen.DSP48E2_inst/DSP_A_B_DATA_INST/A[*]"]


#chain 3
set_multicycle_path -setup 2 -end \
-from [get_pins "gen_dpu.m_dpu_*/m_conv_top/inst_pea/GenOcp[*].u_pe/GenMultCvb.GenMultech[0].GenMidCvb.u_conv_block/GenConvChain[3].inst_dly_tdata/d_r_reg[1][*]*/C"] \
-to   [get_pins "gen_dpu.m_dpu_*/m_conv_top/inst_pea/GenOcp[*].u_pe/GenMultCvb.GenMultech[0].GenMidCvb.u_conv_block/GenConvChain[3].pe_tdata_grp_dly_odd_reg[*]/D"]
set_multicycle_path -hold 1 -end \
-from [get_pins "gen_dpu.m_dpu_*/m_conv_top/inst_pea/GenOcp[*].u_pe/GenMultCvb.GenMultech[0].GenMidCvb.u_conv_block/GenConvChain[3].inst_dly_tdata/d_r_reg[1][*]*/C"] \
-to   [get_pins "gen_dpu.m_dpu_*/m_conv_top/inst_pea/GenOcp[*].u_pe/GenMultCvb.GenMultech[0].GenMidCvb.u_conv_block/GenConvChain[3].pe_tdata_grp_dly_odd_reg[*]/D"]

set_multicycle_path -setup 2 -end \
-from [get_pins "gen_dpu.m_dpu_*/m_conv_top/inst_pea/GenOcp[*].u_pe/GenMultCvb.GenMultech[1].GenLastCvb.u_conv_block/GenConvChain[3].inst_dly_tdata/d_r_reg[1][*]*/C"] \
-to   [get_pins "gen_dpu.m_dpu_*/m_conv_top/inst_pea/GenOcp[*].u_pe/GenMultCvb.GenMultech[1].GenLastCvb.u_conv_block/GenConvChain[3].pe_tdata_grp_dly_odd_reg[*]/D"]
set_multicycle_path -hold 1 -end \
-from [get_pins "gen_dpu.m_dpu_*/m_conv_top/inst_pea/GenOcp[*].u_pe/GenMultCvb.GenMultech[1].GenLastCvb.u_conv_block/GenConvChain[3].inst_dly_tdata/d_r_reg[1][*]*/C"] \
-to   [get_pins "gen_dpu.m_dpu_*/m_conv_top/inst_pea/GenOcp[*].u_pe/GenMultCvb.GenMultech[1].GenLastCvb.u_conv_block/GenConvChain[3].pe_tdata_grp_dly_odd_reg[*]/D"]

#chain4
set_multicycle_path -setup 2 -end \
-from [get_pins "gen_dpu.m_dpu_*/m_conv_top/inst_pea/GenOcp[*].u_pe/GenMultCvb.GenMultech[0].GenMidCvb.u_conv_block/GenConvChain[4].inst_dly_tdata/d_r_reg[2][*]*/C"] \
-to   [get_pins "gen_dpu.m_dpu_*/m_conv_top/inst_pea/GenOcp[*].u_pe/GenMultCvb.GenMultech[0].GenMidCvb.u_conv_block/GenConvChain[4].GenConvPair[*].Gen8x8.GenMultDsp.GenMidDsp.u_dsp_unit_mid/DSP48E2_gen.DSP48E2_inst/DSP_PREADD_DATA_INST/DIN[*]"]
set_multicycle_path -hold 1 -end \
-from [get_pins "gen_dpu.m_dpu_*/m_conv_top/inst_pea/GenOcp[*].u_pe/GenMultCvb.GenMultech[0].GenMidCvb.u_conv_block/GenConvChain[4].inst_dly_tdata/d_r_reg[2][*]*/C"] \
-to   [get_pins "gen_dpu.m_dpu_*/m_conv_top/inst_pea/GenOcp[*].u_pe/GenMultCvb.GenMultech[0].GenMidCvb.u_conv_block/GenConvChain[4].GenConvPair[*].Gen8x8.GenMultDsp.GenMidDsp.u_dsp_unit_mid/DSP48E2_gen.DSP48E2_inst/DSP_PREADD_DATA_INST/DIN[*]"]

set_multicycle_path -setup 2 -end \
-from [get_pins "gen_dpu.m_dpu_*/m_conv_top/inst_pea/GenOcp[*].u_pe/GenMultCvb.GenMultech[0].GenMidCvb.u_conv_block/GenConvChain[4].inst_dly_tdata/d_r_reg[2][*]*/C"] \
-to   [get_pins "gen_dpu.m_dpu_*/m_conv_top/inst_pea/GenOcp[*].u_pe/GenMultCvb.GenMultech[0].GenMidCvb.u_conv_block/GenConvChain[4].GenConvPair[*].Gen8x8.GenMultDsp.GenMidDsp.u_dsp_unit_mid/DSP48E2_gen.DSP48E2_inst/DSP_A_B_DATA_INST/A[*]"]
set_multicycle_path -hold 1 -end \
-from [get_pins "gen_dpu.m_dpu_*/m_conv_top/inst_pea/GenOcp[*].u_pe/GenMultCvb.GenMultech[0].GenMidCvb.u_conv_block/GenConvChain[4].inst_dly_tdata/d_r_reg[2][*]*/C"] \
-to   [get_pins "gen_dpu.m_dpu_*/m_conv_top/inst_pea/GenOcp[*].u_pe/GenMultCvb.GenMultech[0].GenMidCvb.u_conv_block/GenConvChain[4].GenConvPair[*].Gen8x8.GenMultDsp.GenMidDsp.u_dsp_unit_mid/DSP48E2_gen.DSP48E2_inst/DSP_A_B_DATA_INST/A[*]"]

set_multicycle_path -setup 2 -end \
-from [get_pins "gen_dpu.m_dpu_*/m_conv_top/inst_pea/GenOcp[*].u_pe/GenMultCvb.GenMultech[1].GenLastCvb.u_conv_block/GenConvChain[4].inst_dly_tdata/d_r_reg[2][*]*/C"] \
-to   [get_pins "gen_dpu.m_dpu_*/m_conv_top/inst_pea/GenOcp[*].u_pe/GenMultCvb.GenMultech[1].GenLastCvb.u_conv_block/GenConvChain[4].GenConvPair[*].Gen8x8.GenMultDsp.GenMidDsp.u_dsp_unit_mid/DSP48E2_gen.DSP48E2_inst/DSP_PREADD_DATA_INST/DIN[*]"]
set_multicycle_path -hold 1 -end \
-from [get_pins "gen_dpu.m_dpu_*/m_conv_top/inst_pea/GenOcp[*].u_pe/GenMultCvb.GenMultech[1].GenLastCvb.u_conv_block/GenConvChain[4].inst_dly_tdata/d_r_reg[2][*]*/C"] \
-to   [get_pins "gen_dpu.m_dpu_*/m_conv_top/inst_pea/GenOcp[*].u_pe/GenMultCvb.GenMultech[1].GenLastCvb.u_conv_block/GenConvChain[4].GenConvPair[*].Gen8x8.GenMultDsp.GenMidDsp.u_dsp_unit_mid/DSP48E2_gen.DSP48E2_inst/DSP_PREADD_DATA_INST/DIN[*]"]

set_multicycle_path -setup 2 -end \
-from [get_pins "gen_dpu.m_dpu_*/m_conv_top/inst_pea/GenOcp[*].u_pe/GenMultCvb.GenMultech[1].GenLastCvb.u_conv_block/GenConvChain[4].inst_dly_tdata/d_r_reg[2][*]*/C"] \
-to   [get_pins "gen_dpu.m_dpu_*/m_conv_top/inst_pea/GenOcp[*].u_pe/GenMultCvb.GenMultech[1].GenLastCvb.u_conv_block/GenConvChain[4].GenConvPair[*].Gen8x8.GenMultDsp.GenMidDsp.u_dsp_unit_mid/DSP48E2_gen.DSP48E2_inst/DSP_A_B_DATA_INST/A[*]"]
set_multicycle_path -hold 1 -end \
-from [get_pins "gen_dpu.m_dpu_*/m_conv_top/inst_pea/GenOcp[*].u_pe/GenMultCvb.GenMultech[1].GenLastCvb.u_conv_block/GenConvChain[4].inst_dly_tdata/d_r_reg[2][*]*/C"] \
-to   [get_pins "gen_dpu.m_dpu_*/m_conv_top/inst_pea/GenOcp[*].u_pe/GenMultCvb.GenMultech[1].GenLastCvb.u_conv_block/GenConvChain[4].GenConvPair[*].Gen8x8.GenMultDsp.GenMidDsp.u_dsp_unit_mid/DSP48E2_gen.DSP48E2_inst/DSP_A_B_DATA_INST/A[*]"]

#chain 5
set_multicycle_path -setup 2 -end \
-from [get_pins "gen_dpu.m_dpu_*/m_conv_top/inst_pea/GenOcp[*].u_pe/GenMultCvb.GenMultech[0].GenMidCvb.u_conv_block/GenConvChain[5].inst_dly_tdata/d_r_reg[2][*]*/C"] \
-to   [get_pins "gen_dpu.m_dpu_*/m_conv_top/inst_pea/GenOcp[*].u_pe/GenMultCvb.GenMultech[0].GenMidCvb.u_conv_block/GenConvChain[5].pe_tdata_grp_dly_odd_reg[*]/D"]
set_multicycle_path -hold 1 -end \
-from [get_pins "gen_dpu.m_dpu_*/m_conv_top/inst_pea/GenOcp[*].u_pe/GenMultCvb.GenMultech[0].GenMidCvb.u_conv_block/GenConvChain[5].inst_dly_tdata/d_r_reg[2][*]*/C"] \
-to   [get_pins "gen_dpu.m_dpu_*/m_conv_top/inst_pea/GenOcp[*].u_pe/GenMultCvb.GenMultech[0].GenMidCvb.u_conv_block/GenConvChain[5].pe_tdata_grp_dly_odd_reg[*]/D"]

set_multicycle_path -setup 2 -end \
-from [get_pins "gen_dpu.m_dpu_*/m_conv_top/inst_pea/GenOcp[*].u_pe/GenMultCvb.GenMultech[1].GenLastCvb.u_conv_block/GenConvChain[5].inst_dly_tdata/d_r_reg[2][*]*/C"] \
-to   [get_pins "gen_dpu.m_dpu_*/m_conv_top/inst_pea/GenOcp[*].u_pe/GenMultCvb.GenMultech[1].GenLastCvb.u_conv_block/GenConvChain[5].pe_tdata_grp_dly_odd_reg[*]/D"]
set_multicycle_path -hold 1 -end \
-from [get_pins "gen_dpu.m_dpu_*/m_conv_top/inst_pea/GenOcp[*].u_pe/GenMultCvb.GenMultech[1].GenLastCvb.u_conv_block/GenConvChain[5].inst_dly_tdata/d_r_reg[2][*]*/C"] \
-to   [get_pins "gen_dpu.m_dpu_*/m_conv_top/inst_pea/GenOcp[*].u_pe/GenMultCvb.GenMultech[1].GenLastCvb.u_conv_block/GenConvChain[5].pe_tdata_grp_dly_odd_reg[*]/D"]

#chain6
set_multicycle_path -setup 2 -end \
-from [get_pins "gen_dpu.m_dpu_*/m_conv_top/inst_pea/GenOcp[*].u_pe/GenMultCvb.GenMultech[0].GenMidCvb.u_conv_block/GenConvChain[6].inst_dly_tdata/d_r_reg[3][*]*/C"] \
-to   [get_pins "gen_dpu.m_dpu_*/m_conv_top/inst_pea/GenOcp[*].u_pe/GenMultCvb.GenMultech[0].GenMidCvb.u_conv_block/GenConvChain[6].GenConvPair[*].Gen8x8.GenMultDsp.GenMidDsp.u_dsp_unit_mid/DSP48E2_gen.DSP48E2_inst/DSP_PREADD_DATA_INST/DIN[*]"]
set_multicycle_path -hold 1 -end \
-from [get_pins "gen_dpu.m_dpu_*/m_conv_top/inst_pea/GenOcp[*].u_pe/GenMultCvb.GenMultech[0].GenMidCvb.u_conv_block/GenConvChain[6].inst_dly_tdata/d_r_reg[3][*]*/C"] \
-to   [get_pins "gen_dpu.m_dpu_*/m_conv_top/inst_pea/GenOcp[*].u_pe/GenMultCvb.GenMultech[0].GenMidCvb.u_conv_block/GenConvChain[6].GenConvPair[*].Gen8x8.GenMultDsp.GenMidDsp.u_dsp_unit_mid/DSP48E2_gen.DSP48E2_inst/DSP_PREADD_DATA_INST/DIN[*]"]

set_multicycle_path -setup 2 -end \
-from [get_pins "gen_dpu.m_dpu_*/m_conv_top/inst_pea/GenOcp[*].u_pe/GenMultCvb.GenMultech[0].GenMidCvb.u_conv_block/GenConvChain[6].inst_dly_tdata/d_r_reg[3][*]*/C"] \
-to   [get_pins "gen_dpu.m_dpu_*/m_conv_top/inst_pea/GenOcp[*].u_pe/GenMultCvb.GenMultech[0].GenMidCvb.u_conv_block/GenConvChain[6].GenConvPair[*].Gen8x8.GenMultDsp.GenMidDsp.u_dsp_unit_mid/DSP48E2_gen.DSP48E2_inst/DSP_A_B_DATA_INST/A[*]"]
set_multicycle_path -hold 1 -end \
-from [get_pins "gen_dpu.m_dpu_*/m_conv_top/inst_pea/GenOcp[*].u_pe/GenMultCvb.GenMultech[0].GenMidCvb.u_conv_block/GenConvChain[6].inst_dly_tdata/d_r_reg[3][*]*/C"] \
-to   [get_pins "gen_dpu.m_dpu_*/m_conv_top/inst_pea/GenOcp[*].u_pe/GenMultCvb.GenMultech[0].GenMidCvb.u_conv_block/GenConvChain[6].GenConvPair[*].Gen8x8.GenMultDsp.GenMidDsp.u_dsp_unit_mid/DSP48E2_gen.DSP48E2_inst/DSP_A_B_DATA_INST/A[*]"]

set_multicycle_path -setup 2 -end \
-from [get_pins "gen_dpu.m_dpu_*/m_conv_top/inst_pea/GenOcp[*].u_pe/GenMultCvb.GenMultech[1].GenLastCvb.u_conv_block/GenConvChain[6].inst_dly_tdata/d_r_reg[3][*]*/C"] \
-to   [get_pins "gen_dpu.m_dpu_*/m_conv_top/inst_pea/GenOcp[*].u_pe/GenMultCvb.GenMultech[1].GenLastCvb.u_conv_block/GenConvChain[6].GenConvPair[*].Gen8x8.GenMultDsp.GenMidDsp.u_dsp_unit_mid/DSP48E2_gen.DSP48E2_inst/DSP_PREADD_DATA_INST/DIN[*]"]
set_multicycle_path -hold 1 -end \
-from [get_pins "gen_dpu.m_dpu_*/m_conv_top/inst_pea/GenOcp[*].u_pe/GenMultCvb.GenMultech[1].GenLastCvb.u_conv_block/GenConvChain[6].inst_dly_tdata/d_r_reg[3][*]*/C"] \
-to   [get_pins "gen_dpu.m_dpu_*/m_conv_top/inst_pea/GenOcp[*].u_pe/GenMultCvb.GenMultech[1].GenLastCvb.u_conv_block/GenConvChain[6].GenConvPair[*].Gen8x8.GenMultDsp.GenMidDsp.u_dsp_unit_mid/DSP48E2_gen.DSP48E2_inst/DSP_PREADD_DATA_INST/DIN[*]"]

set_multicycle_path -setup 2 -end \
-from [get_pins "gen_dpu.m_dpu_*/m_conv_top/inst_pea/GenOcp[*].u_pe/GenMultCvb.GenMultech[1].GenLastCvb.u_conv_block/GenConvChain[6].inst_dly_tdata/d_r_reg[3][*]*/C"] \
-to   [get_pins "gen_dpu.m_dpu_*/m_conv_top/inst_pea/GenOcp[*].u_pe/GenMultCvb.GenMultech[1].GenLastCvb.u_conv_block/GenConvChain[6].GenConvPair[*].Gen8x8.GenMultDsp.GenMidDsp.u_dsp_unit_mid/DSP48E2_gen.DSP48E2_inst/DSP_A_B_DATA_INST/A[*]"]
set_multicycle_path -hold 1 -end \
-from [get_pins "gen_dpu.m_dpu_*/m_conv_top/inst_pea/GenOcp[*].u_pe/GenMultCvb.GenMultech[1].GenLastCvb.u_conv_block/GenConvChain[6].inst_dly_tdata/d_r_reg[3][*]*/C"] \
-to   [get_pins "gen_dpu.m_dpu_*/m_conv_top/inst_pea/GenOcp[*].u_pe/GenMultCvb.GenMultech[1].GenLastCvb.u_conv_block/GenConvChain[6].GenConvPair[*].Gen8x8.GenMultDsp.GenMidDsp.u_dsp_unit_mid/DSP48E2_gen.DSP48E2_inst/DSP_A_B_DATA_INST/A[*]"]

#chain 7
set_multicycle_path -setup 2 -end \
-from [get_pins "gen_dpu.m_dpu_*/m_conv_top/inst_pea/GenOcp[*].u_pe/GenMultCvb.GenMultech[0].GenMidCvb.u_conv_block/GenConvChain[7].inst_dly_tdata/d_r_reg[3][*]*/C"] \
-to   [get_pins "gen_dpu.m_dpu_*/m_conv_top/inst_pea/GenOcp[*].u_pe/GenMultCvb.GenMultech[0].GenMidCvb.u_conv_block/GenConvChain[7].pe_tdata_grp_dly_odd_reg[*]/D"]
set_multicycle_path -hold 1 -end \
-from [get_pins "gen_dpu.m_dpu_*/m_conv_top/inst_pea/GenOcp[*].u_pe/GenMultCvb.GenMultech[0].GenMidCvb.u_conv_block/GenConvChain[7].inst_dly_tdata/d_r_reg[3][*]*/C"] \
-to   [get_pins "gen_dpu.m_dpu_*/m_conv_top/inst_pea/GenOcp[*].u_pe/GenMultCvb.GenMultech[0].GenMidCvb.u_conv_block/GenConvChain[7].pe_tdata_grp_dly_odd_reg[*]/D"]

set_multicycle_path -setup 2 -end \
-from [get_pins "gen_dpu.m_dpu_*/m_conv_top/inst_pea/GenOcp[*].u_pe/GenMultCvb.GenMultech[1].GenLastCvb.u_conv_block/GenConvChain[7].inst_dly_tdata/d_r_reg[3][*]*/C"] \
-to   [get_pins "gen_dpu.m_dpu_*/m_conv_top/inst_pea/GenOcp[*].u_pe/GenMultCvb.GenMultech[1].GenLastCvb.u_conv_block/GenConvChain[7].pe_tdata_grp_dly_odd_reg[*]/D"]
set_multicycle_path -hold 1 -end \
-from [get_pins "gen_dpu.m_dpu_*/m_conv_top/inst_pea/GenOcp[*].u_pe/GenMultCvb.GenMultech[1].GenLastCvb.u_conv_block/GenConvChain[7].inst_dly_tdata/d_r_reg[3][*]*/C"] \
-to   [get_pins "gen_dpu.m_dpu_*/m_conv_top/inst_pea/GenOcp[*].u_pe/GenMultCvb.GenMultech[1].GenLastCvb.u_conv_block/GenConvChain[7].pe_tdata_grp_dly_odd_reg[*]/D"]

#Weights chain0
set_multicycle_path -setup 2 -end \
-from [get_pins "gen_dpu.m_dpu_*/m_conv_top/inst_pea/GenOcp[*].u_pe/GenMultCvb.GenMultech[*].GenMidCvb.u_conv_block/GenConvChain[0].inst_dly_wgt_0/d_r_reg[0][*]*/C"] \
-to   [get_pins "gen_dpu.m_dpu_*/m_conv_top/inst_pea/GenOcp[*].u_pe/GenMultCvb.GenMultech[*].GenMidCvb.u_conv_block/GenConvChain[0]*/D"]
set_multicycle_path -hold 1 -end \
-from [get_pins "gen_dpu.m_dpu_*/m_conv_top/inst_pea/GenOcp[*].u_pe/GenMultCvb.GenMultech[*].GenMidCvb.u_conv_block/GenConvChain[0].inst_dly_wgt_0/d_r_reg[0][*]*/C"] \
-to   [get_pins "gen_dpu.m_dpu_*/m_conv_top/inst_pea/GenOcp[*].u_pe/GenMultCvb.GenMultech[*].GenMidCvb.u_conv_block/GenConvChain[0]*/D"]
set_multicycle_path -setup 2 -end \
-from [get_pins "gen_dpu.m_dpu_*/m_conv_top/inst_pea/GenOcp[*].u_pe/GenMultCvb.GenMultech[*].GenLastCvb.u_conv_block/GenConvChain[0].inst_dly_wgt_0/d_r_reg[0][*]*/C"] \
-to   [get_pins "gen_dpu.m_dpu_*/m_conv_top/inst_pea/GenOcp[*].u_pe/GenMultCvb.GenMultech[*].GenLastCvb.u_conv_block/GenConvChain[0]*/D"]
set_multicycle_path -hold 1 -end \
-from [get_pins "gen_dpu.m_dpu_*/m_conv_top/inst_pea/GenOcp[*].u_pe/GenMultCvb.GenMultech[*].GenLastCvb.u_conv_block/GenConvChain[0].inst_dly_wgt_0/d_r_reg[0][*]*/C"] \
-to   [get_pins "gen_dpu.m_dpu_*/m_conv_top/inst_pea/GenOcp[*].u_pe/GenMultCvb.GenMultech[*].GenLastCvb.u_conv_block/GenConvChain[0]*/D"]

set_multicycle_path -setup 2 -end \
-from [get_pins "gen_dpu.m_dpu_*/m_conv_top/inst_pea/GenOcp[*].u_pe/GenMultCvb.GenMultech[*].GenMidCvb.u_conv_block/GenConvChain[0].inst_dly_wgt_1/BIT[0].D0.d_r_reg[0][*]*/C"] \
-to   [get_pins "gen_dpu.m_dpu_*/m_conv_top/inst_pea/GenOcp[*].u_pe/GenMultCvb.GenMultech[*].GenMidCvb.u_conv_block/GenConvChain[0]*/D"]
set_multicycle_path -hold 1 -end \
-from [get_pins "gen_dpu.m_dpu_*/m_conv_top/inst_pea/GenOcp[*].u_pe/GenMultCvb.GenMultech[*].GenMidCvb.u_conv_block/GenConvChain[0].inst_dly_wgt_1/BIT[0].D0.d_r_reg[0][*]*/C"] \
-to   [get_pins "gen_dpu.m_dpu_*/m_conv_top/inst_pea/GenOcp[*].u_pe/GenMultCvb.GenMultech[*].GenMidCvb.u_conv_block/GenConvChain[0]*/D"]
set_multicycle_path -setup 2 -end \
-from [get_pins "gen_dpu.m_dpu_*/m_conv_top/inst_pea/GenOcp[*].u_pe/GenMultCvb.GenMultech[*].GenLastCvb.u_conv_block/GenConvChain[0].inst_dly_wgt_1/BIT[0].D0.d_r_reg[0][*]*/C"] \
-to   [get_pins "gen_dpu.m_dpu_*/m_conv_top/inst_pea/GenOcp[*].u_pe/GenMultCvb.GenMultech[*].GenLastCvb.u_conv_block/GenConvChain[0]*/D"]
set_multicycle_path -hold 1 -end \
-from [get_pins "gen_dpu.m_dpu_*/m_conv_top/inst_pea/GenOcp[*].u_pe/GenMultCvb.GenMultech[*].GenLastCvb.u_conv_block/GenConvChain[0].inst_dly_wgt_1/BIT[0].D0.d_r_reg[0][*]*/C"] \
-to   [get_pins "gen_dpu.m_dpu_*/m_conv_top/inst_pea/GenOcp[*].u_pe/GenMultCvb.GenMultech[*].GenLastCvb.u_conv_block/GenConvChain[0]*/D"]

#Weights chain1
set_multicycle_path -setup 2 -end \
-from [get_pins "gen_dpu.m_dpu_*/m_conv_top/inst_pea/GenOcp[*].u_pe/GenMultCvb.GenMultech[*].GenMidCvb.u_conv_block/GenConvChain[1].inst_dly_wgt_0/d_r_reg[0][*]*/C"] \
-to   [get_pins "gen_dpu.m_dpu_*/m_conv_top/inst_pea/GenOcp[*].u_pe/GenMultCvb.GenMultech[*].GenMidCvb.u_conv_block/GenConvChain[1]*/D"]
set_multicycle_path -hold 1 -end \
-from [get_pins "gen_dpu.m_dpu_*/m_conv_top/inst_pea/GenOcp[*].u_pe/GenMultCvb.GenMultech[*].GenMidCvb.u_conv_block/GenConvChain[1].inst_dly_wgt_0/d_r_reg[0][*]*/C"] \
-to   [get_pins "gen_dpu.m_dpu_*/m_conv_top/inst_pea/GenOcp[*].u_pe/GenMultCvb.GenMultech[*].GenMidCvb.u_conv_block/GenConvChain[1]*/D"]
set_multicycle_path -setup 2 -end \
-from [get_pins "gen_dpu.m_dpu_*/m_conv_top/inst_pea/GenOcp[*].u_pe/GenMultCvb.GenMultech[*].GenLastCvb.u_conv_block/GenConvChain[1].inst_dly_wgt_0/d_r_reg[0][*]*/C"] \
-to   [get_pins "gen_dpu.m_dpu_*/m_conv_top/inst_pea/GenOcp[*].u_pe/GenMultCvb.GenMultech[*].GenLastCvb.u_conv_block/GenConvChain[1]*/D"]
set_multicycle_path -hold 1 -end \
-from [get_pins "gen_dpu.m_dpu_*/m_conv_top/inst_pea/GenOcp[*].u_pe/GenMultCvb.GenMultech[*].GenLastCvb.u_conv_block/GenConvChain[1].inst_dly_wgt_0/d_r_reg[0][*]*/C"] \
-to   [get_pins "gen_dpu.m_dpu_*/m_conv_top/inst_pea/GenOcp[*].u_pe/GenMultCvb.GenMultech[*].GenLastCvb.u_conv_block/GenConvChain[1]*/D"]

set_multicycle_path -setup 2 -end \
-from [get_pins "gen_dpu.m_dpu_*/m_conv_top/inst_pea/GenOcp[*].u_pe/GenMultCvb.GenMultech[*].GenMidCvb.u_conv_block/GenConvChain[1].inst_dly_wgt_1/BIT[0].D0.d_r_reg[0][*]*/C"] \
-to   [get_pins "gen_dpu.m_dpu_*/m_conv_top/inst_pea/GenOcp[*].u_pe/GenMultCvb.GenMultech[*].GenMidCvb.u_conv_block/GenConvChain[1]*/D"]
set_multicycle_path -hold 1 -end \
-from [get_pins "gen_dpu.m_dpu_*/m_conv_top/inst_pea/GenOcp[*].u_pe/GenMultCvb.GenMultech[*].GenMidCvb.u_conv_block/GenConvChain[1].inst_dly_wgt_1/BIT[0].D0.d_r_reg[0][*]*/C"] \
-to   [get_pins "gen_dpu.m_dpu_*/m_conv_top/inst_pea/GenOcp[*].u_pe/GenMultCvb.GenMultech[*].GenMidCvb.u_conv_block/GenConvChain[1]*/D"]
set_multicycle_path -setup 2 -end \
-from [get_pins "gen_dpu.m_dpu_*/m_conv_top/inst_pea/GenOcp[*].u_pe/GenMultCvb.GenMultech[*].GenLastCvb.u_conv_block/GenConvChain[1].inst_dly_wgt_1/BIT[0].D0.d_r_reg[0][*]*/C"] \
-to   [get_pins "gen_dpu.m_dpu_*/m_conv_top/inst_pea/GenOcp[*].u_pe/GenMultCvb.GenMultech[*].GenLastCvb.u_conv_block/GenConvChain[1]*/D"]
set_multicycle_path -hold 1 -end \
-from [get_pins "gen_dpu.m_dpu_*/m_conv_top/inst_pea/GenOcp[*].u_pe/GenMultCvb.GenMultech[*].GenLastCvb.u_conv_block/GenConvChain[1].inst_dly_wgt_1/BIT[0].D0.d_r_reg[0][*]*/C"] \
-to   [get_pins "gen_dpu.m_dpu_*/m_conv_top/inst_pea/GenOcp[*].u_pe/GenMultCvb.GenMultech[*].GenLastCvb.u_conv_block/GenConvChain[1]*/D"]

#Weights chain2
set_multicycle_path -setup 2 -end \
-from [get_pins "gen_dpu.m_dpu_*/m_conv_top/inst_pea/GenOcp[*].u_pe/GenMultCvb.GenMultech[*].GenMidCvb.u_conv_block/GenConvChain[2].inst_dly_wgt_0/d_r_reg[1][*]*/C"] \
-to   [get_pins "gen_dpu.m_dpu_*/m_conv_top/inst_pea/GenOcp[*].u_pe/GenMultCvb.GenMultech[*].GenMidCvb.u_conv_block/GenConvChain[2]*/D"]
set_multicycle_path -hold 1 -end \
-from [get_pins "gen_dpu.m_dpu_*/m_conv_top/inst_pea/GenOcp[*].u_pe/GenMultCvb.GenMultech[*].GenMidCvb.u_conv_block/GenConvChain[2].inst_dly_wgt_0/d_r_reg[1][*]*/C"] \
-to   [get_pins "gen_dpu.m_dpu_*/m_conv_top/inst_pea/GenOcp[*].u_pe/GenMultCvb.GenMultech[*].GenMidCvb.u_conv_block/GenConvChain[2]*/D"]
set_multicycle_path -setup 2 -end \
-from [get_pins "gen_dpu.m_dpu_*/m_conv_top/inst_pea/GenOcp[*].u_pe/GenMultCvb.GenMultech[*].GenLastCvb.u_conv_block/GenConvChain[2].inst_dly_wgt_0/d_r_reg[1][*]*/C"] \
-to   [get_pins "gen_dpu.m_dpu_*/m_conv_top/inst_pea/GenOcp[*].u_pe/GenMultCvb.GenMultech[*].GenLastCvb.u_conv_block/GenConvChain[2]*/D"]
set_multicycle_path -hold 1 -end \
-from [get_pins "gen_dpu.m_dpu_*/m_conv_top/inst_pea/GenOcp[*].u_pe/GenMultCvb.GenMultech[*].GenLastCvb.u_conv_block/GenConvChain[2].inst_dly_wgt_0/d_r_reg[1][*]*/C"] \
-to   [get_pins "gen_dpu.m_dpu_*/m_conv_top/inst_pea/GenOcp[*].u_pe/GenMultCvb.GenMultech[*].GenLastCvb.u_conv_block/GenConvChain[2]*/D"]

set_multicycle_path -setup 2 -end \
-from [get_pins "gen_dpu.m_dpu_*/m_conv_top/inst_pea/GenOcp[*].u_pe/GenMultCvb.GenMultech[*].GenMidCvb.u_conv_block/GenConvChain[2].inst_dly_wgt_1/BIT[1].DX.d_r_reg[1][*]*/C"] \
-to   [get_pins "gen_dpu.m_dpu_*/m_conv_top/inst_pea/GenOcp[*].u_pe/GenMultCvb.GenMultech[*].GenMidCvb.u_conv_block/GenConvChain[2]*/D"]
set_multicycle_path -hold 1 -end \
-from [get_pins "gen_dpu.m_dpu_*/m_conv_top/inst_pea/GenOcp[*].u_pe/GenMultCvb.GenMultech[*].GenMidCvb.u_conv_block/GenConvChain[2].inst_dly_wgt_1/BIT[1].DX.d_r_reg[1][*]*/C"] \
-to   [get_pins "gen_dpu.m_dpu_*/m_conv_top/inst_pea/GenOcp[*].u_pe/GenMultCvb.GenMultech[*].GenMidCvb.u_conv_block/GenConvChain[2]*/D"]
set_multicycle_path -setup 2 -end \
-from [get_pins "gen_dpu.m_dpu_*/m_conv_top/inst_pea/GenOcp[*].u_pe/GenMultCvb.GenMultech[*].GenLastCvb.u_conv_block/GenConvChain[2].inst_dly_wgt_1/BIT[1].DX.d_r_reg[1][*]*/C"] \
-to   [get_pins "gen_dpu.m_dpu_*/m_conv_top/inst_pea/GenOcp[*].u_pe/GenMultCvb.GenMultech[*].GenLastCvb.u_conv_block/GenConvChain[2]*/D"]
set_multicycle_path -hold 1 -end \
-from [get_pins "gen_dpu.m_dpu_*/m_conv_top/inst_pea/GenOcp[*].u_pe/GenMultCvb.GenMultech[*].GenLastCvb.u_conv_block/GenConvChain[2].inst_dly_wgt_1/BIT[1].DX.d_r_reg[1][*]*/C"] \
-to   [get_pins "gen_dpu.m_dpu_*/m_conv_top/inst_pea/GenOcp[*].u_pe/GenMultCvb.GenMultech[*].GenLastCvb.u_conv_block/GenConvChain[2]*/D"]


#Weights chain3
set_multicycle_path -setup 2 -end \
-from [get_pins "gen_dpu.m_dpu_*/m_conv_top/inst_pea/GenOcp[*].u_pe/GenMultCvb.GenMultech[*].GenMidCvb.u_conv_block/GenConvChain[3].inst_dly_wgt_0/d_r_reg[1][*]*/C"] \
-to   [get_pins "gen_dpu.m_dpu_*/m_conv_top/inst_pea/GenOcp[*].u_pe/GenMultCvb.GenMultech[*].GenMidCvb.u_conv_block/GenConvChain[3]*/D"]
set_multicycle_path -hold 1 -end \
-from [get_pins "gen_dpu.m_dpu_*/m_conv_top/inst_pea/GenOcp[*].u_pe/GenMultCvb.GenMultech[*].GenMidCvb.u_conv_block/GenConvChain[3].inst_dly_wgt_0/d_r_reg[1][*]*/C"] \
-to   [get_pins "gen_dpu.m_dpu_*/m_conv_top/inst_pea/GenOcp[*].u_pe/GenMultCvb.GenMultech[*].GenMidCvb.u_conv_block/GenConvChain[3]*/D"]
set_multicycle_path -setup 2 -end \
-from [get_pins "gen_dpu.m_dpu_*/m_conv_top/inst_pea/GenOcp[*].u_pe/GenMultCvb.GenMultech[*].GenLastCvb.u_conv_block/GenConvChain[3].inst_dly_wgt_0/d_r_reg[1][*]*/C"] \
-to   [get_pins "gen_dpu.m_dpu_*/m_conv_top/inst_pea/GenOcp[*].u_pe/GenMultCvb.GenMultech[*].GenLastCvb.u_conv_block/GenConvChain[3]*/D"]
set_multicycle_path -hold 1 -end \
-from [get_pins "gen_dpu.m_dpu_*/m_conv_top/inst_pea/GenOcp[*].u_pe/GenMultCvb.GenMultech[*].GenLastCvb.u_conv_block/GenConvChain[3].inst_dly_wgt_0/d_r_reg[1][*]*/C"] \
-to   [get_pins "gen_dpu.m_dpu_*/m_conv_top/inst_pea/GenOcp[*].u_pe/GenMultCvb.GenMultech[*].GenLastCvb.u_conv_block/GenConvChain[3]*/D"]

set_multicycle_path -setup 2 -end \
-from [get_pins "gen_dpu.m_dpu_*/m_conv_top/inst_pea/GenOcp[*].u_pe/GenMultCvb.GenMultech[*].GenMidCvb.u_conv_block/GenConvChain[3].inst_dly_wgt_1/BIT[1].DX.d_r_reg[1][*]*/C"] \
-to   [get_pins "gen_dpu.m_dpu_*/m_conv_top/inst_pea/GenOcp[*].u_pe/GenMultCvb.GenMultech[*].GenMidCvb.u_conv_block/GenConvChain[3]*/D"]
set_multicycle_path -hold 1 -end \
-from [get_pins "gen_dpu.m_dpu_*/m_conv_top/inst_pea/GenOcp[*].u_pe/GenMultCvb.GenMultech[*].GenMidCvb.u_conv_block/GenConvChain[3].inst_dly_wgt_1/BIT[1].DX.d_r_reg[1][*]*/C"] \
-to   [get_pins "gen_dpu.m_dpu_*/m_conv_top/inst_pea/GenOcp[*].u_pe/GenMultCvb.GenMultech[*].GenMidCvb.u_conv_block/GenConvChain[3]*/D"]
set_multicycle_path -setup 2 -end \
-from [get_pins "gen_dpu.m_dpu_*/m_conv_top/inst_pea/GenOcp[*].u_pe/GenMultCvb.GenMultech[*].GenLastCvb.u_conv_block/GenConvChain[3].inst_dly_wgt_1/BIT[1].DX.d_r_reg[1][*]*/C"] \
-to   [get_pins "gen_dpu.m_dpu_*/m_conv_top/inst_pea/GenOcp[*].u_pe/GenMultCvb.GenMultech[*].GenLastCvb.u_conv_block/GenConvChain[3]*/D"]
set_multicycle_path -hold 1 -end \
-from [get_pins "gen_dpu.m_dpu_*/m_conv_top/inst_pea/GenOcp[*].u_pe/GenMultCvb.GenMultech[*].GenLastCvb.u_conv_block/GenConvChain[3].inst_dly_wgt_1/BIT[1].DX.d_r_reg[1][*]*/C"] \
-to   [get_pins "gen_dpu.m_dpu_*/m_conv_top/inst_pea/GenOcp[*].u_pe/GenMultCvb.GenMultech[*].GenLastCvb.u_conv_block/GenConvChain[3]*/D"]


#Weights chain4
set_multicycle_path -setup 2 -end \
-from [get_pins "gen_dpu.m_dpu_*/m_conv_top/inst_pea/GenOcp[*].u_pe/GenMultCvb.GenMultech[*].GenMidCvb.u_conv_block/GenConvChain[4].inst_dly_wgt_0/d_r_reg[2][*]*/C"] \
-to   [get_pins "gen_dpu.m_dpu_*/m_conv_top/inst_pea/GenOcp[*].u_pe/GenMultCvb.GenMultech[*].GenMidCvb.u_conv_block/GenConvChain[4]*/D"]
set_multicycle_path -hold 1 -end \
-from [get_pins "gen_dpu.m_dpu_*/m_conv_top/inst_pea/GenOcp[*].u_pe/GenMultCvb.GenMultech[*].GenMidCvb.u_conv_block/GenConvChain[4].inst_dly_wgt_0/d_r_reg[2][*]*/C"] \
-to   [get_pins "gen_dpu.m_dpu_*/m_conv_top/inst_pea/GenOcp[*].u_pe/GenMultCvb.GenMultech[*].GenMidCvb.u_conv_block/GenConvChain[4]*/D"]
set_multicycle_path -setup 2 -end \
-from [get_pins "gen_dpu.m_dpu_*/m_conv_top/inst_pea/GenOcp[*].u_pe/GenMultCvb.GenMultech[*].GenLastCvb.u_conv_block/GenConvChain[4].inst_dly_wgt_0/d_r_reg[2][*]*/C"] \
-to   [get_pins "gen_dpu.m_dpu_*/m_conv_top/inst_pea/GenOcp[*].u_pe/GenMultCvb.GenMultech[*].GenLastCvb.u_conv_block/GenConvChain[4]*/D"]
set_multicycle_path -hold 1 -end \
-from [get_pins "gen_dpu.m_dpu_*/m_conv_top/inst_pea/GenOcp[*].u_pe/GenMultCvb.GenMultech[*].GenLastCvb.u_conv_block/GenConvChain[4].inst_dly_wgt_0/d_r_reg[2][*]*/C"] \
-to   [get_pins "gen_dpu.m_dpu_*/m_conv_top/inst_pea/GenOcp[*].u_pe/GenMultCvb.GenMultech[*].GenLastCvb.u_conv_block/GenConvChain[4]*/D"]

set_multicycle_path -setup 2 -end \
-from [get_pins "gen_dpu.m_dpu_*/m_conv_top/inst_pea/GenOcp[*].u_pe/GenMultCvb.GenMultech[*].GenMidCvb.u_conv_block/GenConvChain[4].inst_dly_wgt_1/BIT[2].DX.d_r_reg[2][*]*/C"] \
-to   [get_pins "gen_dpu.m_dpu_*/m_conv_top/inst_pea/GenOcp[*].u_pe/GenMultCvb.GenMultech[*].GenMidCvb.u_conv_block/GenConvChain[4]*/D"]
set_multicycle_path -hold 1 -end \
-from [get_pins "gen_dpu.m_dpu_*/m_conv_top/inst_pea/GenOcp[*].u_pe/GenMultCvb.GenMultech[*].GenMidCvb.u_conv_block/GenConvChain[4].inst_dly_wgt_1/BIT[2].DX.d_r_reg[2][*]*/C"] \
-to   [get_pins "gen_dpu.m_dpu_*/m_conv_top/inst_pea/GenOcp[*].u_pe/GenMultCvb.GenMultech[*].GenMidCvb.u_conv_block/GenConvChain[4]*/D"]
set_multicycle_path -setup 2 -end \
-from [get_pins "gen_dpu.m_dpu_*/m_conv_top/inst_pea/GenOcp[*].u_pe/GenMultCvb.GenMultech[*].GenLastCvb.u_conv_block/GenConvChain[4].inst_dly_wgt_1/BIT[2].DX.d_r_reg[2][*]*/C"] \
-to   [get_pins "gen_dpu.m_dpu_*/m_conv_top/inst_pea/GenOcp[*].u_pe/GenMultCvb.GenMultech[*].GenLastCvb.u_conv_block/GenConvChain[4]*/D"]
set_multicycle_path -hold 1 -end \
-from [get_pins "gen_dpu.m_dpu_*/m_conv_top/inst_pea/GenOcp[*].u_pe/GenMultCvb.GenMultech[*].GenLastCvb.u_conv_block/GenConvChain[4].inst_dly_wgt_1/BIT[2].DX.d_r_reg[2][*]*/C"] \
-to   [get_pins "gen_dpu.m_dpu_*/m_conv_top/inst_pea/GenOcp[*].u_pe/GenMultCvb.GenMultech[*].GenLastCvb.u_conv_block/GenConvChain[4]*/D"]

#Weights chain5
set_multicycle_path -setup 2 -end \
-from [get_pins "gen_dpu.m_dpu_*/m_conv_top/inst_pea/GenOcp[*].u_pe/GenMultCvb.GenMultech[*].GenMidCvb.u_conv_block/GenConvChain[5].inst_dly_wgt_0/d_r_reg[2][*]*/C"] \
-to   [get_pins "gen_dpu.m_dpu_*/m_conv_top/inst_pea/GenOcp[*].u_pe/GenMultCvb.GenMultech[*].GenMidCvb.u_conv_block/GenConvChain[5]*/D"]
set_multicycle_path -hold 1 -end \
-from [get_pins "gen_dpu.m_dpu_*/m_conv_top/inst_pea/GenOcp[*].u_pe/GenMultCvb.GenMultech[*].GenMidCvb.u_conv_block/GenConvChain[5].inst_dly_wgt_0/d_r_reg[2][*]*/C"] \
-to   [get_pins "gen_dpu.m_dpu_*/m_conv_top/inst_pea/GenOcp[*].u_pe/GenMultCvb.GenMultech[*].GenMidCvb.u_conv_block/GenConvChain[5]*/D"]
set_multicycle_path -setup 2 -end \
-from [get_pins "gen_dpu.m_dpu_*/m_conv_top/inst_pea/GenOcp[*].u_pe/GenMultCvb.GenMultech[*].GenLastCvb.u_conv_block/GenConvChain[5].inst_dly_wgt_0/d_r_reg[2][*]*/C"] \
-to   [get_pins "gen_dpu.m_dpu_*/m_conv_top/inst_pea/GenOcp[*].u_pe/GenMultCvb.GenMultech[*].GenLastCvb.u_conv_block/GenConvChain[5]*/D"]
set_multicycle_path -hold 1 -end \
-from [get_pins "gen_dpu.m_dpu_*/m_conv_top/inst_pea/GenOcp[*].u_pe/GenMultCvb.GenMultech[*].GenLastCvb.u_conv_block/GenConvChain[5].inst_dly_wgt_0/d_r_reg[2][*]*/C"] \
-to   [get_pins "gen_dpu.m_dpu_*/m_conv_top/inst_pea/GenOcp[*].u_pe/GenMultCvb.GenMultech[*].GenLastCvb.u_conv_block/GenConvChain[5]*/D"]

set_multicycle_path -setup 2 -end \
-from [get_pins "gen_dpu.m_dpu_*/m_conv_top/inst_pea/GenOcp[*].u_pe/GenMultCvb.GenMultech[*].GenMidCvb.u_conv_block/GenConvChain[5].inst_dly_wgt_1/BIT[2].DX.d_r_reg[2][*]*/C"] \
-to   [get_pins "gen_dpu.m_dpu_*/m_conv_top/inst_pea/GenOcp[*].u_pe/GenMultCvb.GenMultech[*].GenMidCvb.u_conv_block/GenConvChain[5]*/D"]
set_multicycle_path -hold 1 -end \
-from [get_pins "gen_dpu.m_dpu_*/m_conv_top/inst_pea/GenOcp[*].u_pe/GenMultCvb.GenMultech[*].GenMidCvb.u_conv_block/GenConvChain[5].inst_dly_wgt_1/BIT[2].DX.d_r_reg[2][*]*/C"] \
-to   [get_pins "gen_dpu.m_dpu_*/m_conv_top/inst_pea/GenOcp[*].u_pe/GenMultCvb.GenMultech[*].GenMidCvb.u_conv_block/GenConvChain[5]*/D"]
set_multicycle_path -setup 2 -end \
-from [get_pins "gen_dpu.m_dpu_*/m_conv_top/inst_pea/GenOcp[*].u_pe/GenMultCvb.GenMultech[*].GenLastCvb.u_conv_block/GenConvChain[5].inst_dly_wgt_1/BIT[2].DX.d_r_reg[2][*]*/C"] \
-to   [get_pins "gen_dpu.m_dpu_*/m_conv_top/inst_pea/GenOcp[*].u_pe/GenMultCvb.GenMultech[*].GenLastCvb.u_conv_block/GenConvChain[5]*/D"]
set_multicycle_path -hold 1 -end \
-from [get_pins "gen_dpu.m_dpu_*/m_conv_top/inst_pea/GenOcp[*].u_pe/GenMultCvb.GenMultech[*].GenLastCvb.u_conv_block/GenConvChain[5].inst_dly_wgt_1/BIT[2].DX.d_r_reg[2][*]*/C"] \
-to   [get_pins "gen_dpu.m_dpu_*/m_conv_top/inst_pea/GenOcp[*].u_pe/GenMultCvb.GenMultech[*].GenLastCvb.u_conv_block/GenConvChain[5]*/D"]

#Weights chain6
set_multicycle_path -setup 2 -end \
-from [get_pins "gen_dpu.m_dpu_*/m_conv_top/inst_pea/GenOcp[*].u_pe/GenMultCvb.GenMultech[*].GenMidCvb.u_conv_block/GenConvChain[6].inst_dly_wgt_0/d_r_reg[3][*]*/C"] \
-to   [get_pins "gen_dpu.m_dpu_*/m_conv_top/inst_pea/GenOcp[*].u_pe/GenMultCvb.GenMultech[*].GenMidCvb.u_conv_block/GenConvChain[6]*/D"]
set_multicycle_path -hold 1 -end \
-from [get_pins "gen_dpu.m_dpu_*/m_conv_top/inst_pea/GenOcp[*].u_pe/GenMultCvb.GenMultech[*].GenMidCvb.u_conv_block/GenConvChain[6].inst_dly_wgt_0/d_r_reg[3][*]*/C"] \
-to   [get_pins "gen_dpu.m_dpu_*/m_conv_top/inst_pea/GenOcp[*].u_pe/GenMultCvb.GenMultech[*].GenMidCvb.u_conv_block/GenConvChain[6]*/D"]
set_multicycle_path -setup 2 -end \
-from [get_pins "gen_dpu.m_dpu_*/m_conv_top/inst_pea/GenOcp[*].u_pe/GenMultCvb.GenMultech[*].GenLastCvb.u_conv_block/GenConvChain[6].inst_dly_wgt_0/d_r_reg[3][*]*/C"] \
-to   [get_pins "gen_dpu.m_dpu_*/m_conv_top/inst_pea/GenOcp[*].u_pe/GenMultCvb.GenMultech[*].GenLastCvb.u_conv_block/GenConvChain[6]*/D"]
set_multicycle_path -hold 1 -end \
-from [get_pins "gen_dpu.m_dpu_*/m_conv_top/inst_pea/GenOcp[*].u_pe/GenMultCvb.GenMultech[*].GenLastCvb.u_conv_block/GenConvChain[6].inst_dly_wgt_0/d_r_reg[3][*]*/C"] \
-to   [get_pins "gen_dpu.m_dpu_*/m_conv_top/inst_pea/GenOcp[*].u_pe/GenMultCvb.GenMultech[*].GenLastCvb.u_conv_block/GenConvChain[6]*/D"]

set_multicycle_path -setup 2 -end \
-from [get_pins "gen_dpu.m_dpu_*/m_conv_top/inst_pea/GenOcp[*].u_pe/GenMultCvb.GenMultech[*].GenMidCvb.u_conv_block/GenConvChain[6].inst_dly_wgt_1/BIT[3].DX.d_r_reg[3][*]*/C"] \
-to   [get_pins "gen_dpu.m_dpu_*/m_conv_top/inst_pea/GenOcp[*].u_pe/GenMultCvb.GenMultech[*].GenMidCvb.u_conv_block/GenConvChain[6]*/D"]
set_multicycle_path -hold 1 -end \
-from [get_pins "gen_dpu.m_dpu_*/m_conv_top/inst_pea/GenOcp[*].u_pe/GenMultCvb.GenMultech[*].GenMidCvb.u_conv_block/GenConvChain[6].inst_dly_wgt_1/BIT[3].DX.d_r_reg[3][*]*/C"] \
-to   [get_pins "gen_dpu.m_dpu_*/m_conv_top/inst_pea/GenOcp[*].u_pe/GenMultCvb.GenMultech[*].GenMidCvb.u_conv_block/GenConvChain[6]*/D"]
set_multicycle_path -setup 2 -end \
-from [get_pins "gen_dpu.m_dpu_*/m_conv_top/inst_pea/GenOcp[*].u_pe/GenMultCvb.GenMultech[*].GenLastCvb.u_conv_block/GenConvChain[6].inst_dly_wgt_1/BIT[3].DX.d_r_reg[3][*]*/C"] \
-to   [get_pins "gen_dpu.m_dpu_*/m_conv_top/inst_pea/GenOcp[*].u_pe/GenMultCvb.GenMultech[*].GenLastCvb.u_conv_block/GenConvChain[6]*/D"]
set_multicycle_path -hold 1 -end \
-from [get_pins "gen_dpu.m_dpu_*/m_conv_top/inst_pea/GenOcp[*].u_pe/GenMultCvb.GenMultech[*].GenLastCvb.u_conv_block/GenConvChain[6].inst_dly_wgt_1/BIT[3].DX.d_r_reg[3][*]*/C"] \
-to   [get_pins "gen_dpu.m_dpu_*/m_conv_top/inst_pea/GenOcp[*].u_pe/GenMultCvb.GenMultech[*].GenLastCvb.u_conv_block/GenConvChain[6]*/D"]

#Weights chain7
set_multicycle_path -setup 2 -end \
-from [get_pins "gen_dpu.m_dpu_*/m_conv_top/inst_pea/GenOcp[*].u_pe/GenMultCvb.GenMultech[*].GenMidCvb.u_conv_block/GenConvChain[7].inst_dly_wgt_0/d_r_reg[3][*]*/C"] \
-to   [get_pins "gen_dpu.m_dpu_*/m_conv_top/inst_pea/GenOcp[*].u_pe/GenMultCvb.GenMultech[*].GenMidCvb.u_conv_block/GenConvChain[7]*/D"]
set_multicycle_path -hold 1 -end \
-from [get_pins "gen_dpu.m_dpu_*/m_conv_top/inst_pea/GenOcp[*].u_pe/GenMultCvb.GenMultech[*].GenMidCvb.u_conv_block/GenConvChain[7].inst_dly_wgt_0/d_r_reg[3][*]*/C"] \
-to   [get_pins "gen_dpu.m_dpu_*/m_conv_top/inst_pea/GenOcp[*].u_pe/GenMultCvb.GenMultech[*].GenMidCvb.u_conv_block/GenConvChain[7]*/D"]
set_multicycle_path -setup 2 -end \
-from [get_pins "gen_dpu.m_dpu_*/m_conv_top/inst_pea/GenOcp[*].u_pe/GenMultCvb.GenMultech[*].GenLastCvb.u_conv_block/GenConvChain[7].inst_dly_wgt_0/d_r_reg[3][*]*/C"] \
-to   [get_pins "gen_dpu.m_dpu_*/m_conv_top/inst_pea/GenOcp[*].u_pe/GenMultCvb.GenMultech[*].GenLastCvb.u_conv_block/GenConvChain[7]*/D"]
set_multicycle_path -hold 1 -end \
-from [get_pins "gen_dpu.m_dpu_*/m_conv_top/inst_pea/GenOcp[*].u_pe/GenMultCvb.GenMultech[*].GenLastCvb.u_conv_block/GenConvChain[7].inst_dly_wgt_0/d_r_reg[3][*]*/C"] \
-to   [get_pins "gen_dpu.m_dpu_*/m_conv_top/inst_pea/GenOcp[*].u_pe/GenMultCvb.GenMultech[*].GenLastCvb.u_conv_block/GenConvChain[7]*/D"]

set_multicycle_path -setup 2 -end \
-from [get_pins "gen_dpu.m_dpu_*/m_conv_top/inst_pea/GenOcp[*].u_pe/GenMultCvb.GenMultech[*].GenMidCvb.u_conv_block/GenConvChain[7].inst_dly_wgt_1/BIT[3].DX.d_r_reg[3][*]*/C"] \
-to   [get_pins "gen_dpu.m_dpu_*/m_conv_top/inst_pea/GenOcp[*].u_pe/GenMultCvb.GenMultech[*].GenMidCvb.u_conv_block/GenConvChain[7]*/D"]
set_multicycle_path -hold 1 -end \
-from [get_pins "gen_dpu.m_dpu_*/m_conv_top/inst_pea/GenOcp[*].u_pe/GenMultCvb.GenMultech[*].GenMidCvb.u_conv_block/GenConvChain[7].inst_dly_wgt_1/BIT[3].DX.d_r_reg[3][*]*/C"] \
-to   [get_pins "gen_dpu.m_dpu_*/m_conv_top/inst_pea/GenOcp[*].u_pe/GenMultCvb.GenMultech[*].GenMidCvb.u_conv_block/GenConvChain[7]*/D"]
set_multicycle_path -setup 2 -end \
-from [get_pins "gen_dpu.m_dpu_*/m_conv_top/inst_pea/GenOcp[*].u_pe/GenMultCvb.GenMultech[*].GenLastCvb.u_conv_block/GenConvChain[7].inst_dly_wgt_1/BIT[3].DX.d_r_reg[3][*]*/C"] \
-to   [get_pins "gen_dpu.m_dpu_*/m_conv_top/inst_pea/GenOcp[*].u_pe/GenMultCvb.GenMultech[*].GenLastCvb.u_conv_block/GenConvChain[7]*/D"]
set_multicycle_path -hold 1 -end \
-from [get_pins "gen_dpu.m_dpu_*/m_conv_top/inst_pea/GenOcp[*].u_pe/GenMultCvb.GenMultech[*].GenLastCvb.u_conv_block/GenConvChain[7].inst_dly_wgt_1/BIT[3].DX.d_r_reg[3][*]*/C"] \
-to   [get_pins "gen_dpu.m_dpu_*/m_conv_top/inst_pea/GenOcp[*].u_pe/GenMultCvb.GenMultech[*].GenLastCvb.u_conv_block/GenConvChain[7]*/D"]

#20190105, add bias's path's MCP
set_multicycle_path -setup 4 -from [get_pins "gen_dpu.m_dpu_*/m_conv_top/inst_reader/inst_rd_bias/bias_r_reg[*]/C"] -to  [get_pins "gen_dpu.m_dpu_*/m_conv_top/inst_pea/GenOcp[*].pe_bias0_reg[*]/D"]
set_multicycle_path -hold  3 -from [get_pins "gen_dpu.m_dpu_*/m_conv_top/inst_reader/inst_rd_bias/bias_r_reg[*]/C"] -to  [get_pins "gen_dpu.m_dpu_*/m_conv_top/inst_pea/GenOcp[*].pe_bias0_reg[*]/D"]
set_multicycle_path -setup 4 -from [get_pins "gen_dpu.m_dpu_*/m_conv_top/inst_ins_parser/reg_bias_shf_r_reg[*]/C"]  -to  [get_pins "gen_dpu.m_dpu_*/m_conv_top/inst_pea/GenOcp[*].pe_bias0_reg[*]/D"]
set_multicycle_path -hold  3 -from [get_pins "gen_dpu.m_dpu_*/m_conv_top/inst_ins_parser/reg_bias_shf_r_reg[*]/C"]  -to  [get_pins "gen_dpu.m_dpu_*/m_conv_top/inst_pea/GenOcp[*].pe_bias0_reg[*]/D"]

set_multicycle_path -setup 4 -from [get_pins "gen_dpu.m_dpu_*/m_conv_top/inst_reader/inst_rd_bias/bias_r_reg[*]/C"] -to  [get_pins "gen_dpu.m_dpu_*/m_conv_top/inst_pea/GenOcp[*].pe_bias1_reg[*]/D"]
set_multicycle_path -hold  3 -from [get_pins "gen_dpu.m_dpu_*/m_conv_top/inst_reader/inst_rd_bias/bias_r_reg[*]/C"] -to  [get_pins "gen_dpu.m_dpu_*/m_conv_top/inst_pea/GenOcp[*].pe_bias1_reg[*]/D"]
set_multicycle_path -setup 4 -from [get_pins "gen_dpu.m_dpu_*/m_conv_top/inst_ins_parser/reg_bias_shf_r_reg[*]/C"]  -to  [get_pins "gen_dpu.m_dpu_*/m_conv_top/inst_pea/GenOcp[*].pe_bias1_reg[*]/D"]
set_multicycle_path -hold  3 -from [get_pins "gen_dpu.m_dpu_*/m_conv_top/inst_ins_parser/reg_bias_shf_r_reg[*]/C"]  -to  [get_pins "gen_dpu.m_dpu_*/m_conv_top/inst_pea/GenOcp[*].pe_bias1_reg[*]/D"]

#Temp #2019-02-27, add MCP for channel auth mode selection
#Temp #FIXME: need SunChong to confirm
#Temp #set_multicycle_path -setup 2 -from [get_pins "m_wgt_loader/loader/bank_writer/is_chs_aug_mode_reg/C"]
#Temp #set_multicycle_path -hold  1 -from [get_pins "m_wgt_loader/loader/bank_writer/is_chs_aug_mode_reg/C"]
#Temp #
#Temp #set_multicycle_path -setup 2 -from [get_pins "m_fm_loader/loader*/bank_writer/is_chs_aug_mode_reg/C"]
#Temp #set_multicycle_path -hold  1 -from [get_pins "m_fm_loader/loader*/bank_writer/is_chs_aug_mode_reg/C"]
#Temp 
#Temp #2019-02-27, add MCP for CFG selection in loader
#Temp #FIXME: need SunChong to confirm
#Temp #set_multicycle_path -setup 2 -from [get_pins " \
#Temp #m_wgt_loader/loader/instr_dispatch/cfg_buf_wr_bytes_num_reg[*]/C \
#Temp #m_wgt_loader/loader/instr_dispatch/cfg_jump_wr_reg[*]/C \
#Temp #m_wgt_loader/loader/instr_dispatch/cfg_jump_rd_reg[*]/C \
#Temp #m_wgt_loader/loader/instr_dispatch/cfg_length_reg[*]/C \
#Temp #m_wgt_loader/loader/instr_dispatch/cfg_in_chs_reg[*]/C  \
#Temp #m_wgt_loader/loader/instr_dispatch/cfg_pad_start_reg[*]/C \
#Temp #m_wgt_loader/loader/instr_dispatch/cfg_buf_avg_mode_reg[*]/C \
#Temp #m_wgt_loader/loader/instr_dispatch/cfg_buf_wr_bank_id_reg[*]/C"]
#Temp #set_multicycle_path -hold  1 -from [get_pins " \
#Temp #m_wgt_loader/loader/instr_dispatch/cfg_buf_wr_bytes_num_reg[*]/C \
#Temp #m_wgt_loader/loader/instr_dispatch/cfg_jump_wr_reg[*]/C \
#Temp #m_wgt_loader/loader/instr_dispatch/cfg_jump_rd_reg[*]/C \
#Temp #m_wgt_loader/loader/instr_dispatch/cfg_length_reg[*]/C \
#Temp #m_wgt_loader/loader/instr_dispatch/cfg_in_chs_reg[*]/C  \
#Temp #m_wgt_loader/loader/instr_dispatch/cfg_pad_start_reg[*]/C \
#Temp #m_wgt_loader/loader/instr_dispatch/cfg_buf_avg_mode_reg[*]/C \
#Temp #m_wgt_loader/loader/instr_dispatch/cfg_buf_wr_bank_id_reg[*]/C"]
#Temp #
#Temp #set_multicycle_path -setup 2 -from [get_pins " \
#Temp #m_fm_loader/loader*/instr_dispatch/cfg_buf_wr_bytes_num_reg[*]/C \
#Temp #m_fm_loader/loader*/instr_dispatch/cfg_jump_wr_reg[*]/C \
#Temp #m_fm_loader/loader*/instr_dispatch/cfg_jump_rd_reg[*]/C \
#Temp #m_fm_loader/loader*/instr_dispatch/cfg_length_reg[*]/C \
#Temp #m_fm_loader/loader*/instr_dispatch/cfg_in_chs_reg[*]/C \
#Temp #m_fm_loader/loader*/instr_dispatch/cfg_pad_idx_reg[*]/C \
#Temp #m_fm_loader/loader*/instr_dispatch/cfg_pad_start_reg[*]/C \
#Temp #m_fm_loader/loader*/instr_dispatch/cfg_pad_end_reg[*]/C \
#Temp #m_fm_loader/loader*/instr_dispatch/cfg_buf_avg_mode_reg[*]/C \
#Temp #m_fm_loader/loader*/instr_dispatch/cfg_buf_wr_bank_id_reg[*]/C"]
#Temp #set_multicycle_path -hold  1 -from [get_pins " \
#Temp #m_fm_loader/loader*/instr_dispatch/cfg_buf_wr_bytes_num_reg[*]/C \
#Temp #m_fm_loader/loader*/instr_dispatch/cfg_jump_wr_reg[*]/C \
#Temp #m_fm_loader/loader*/instr_dispatch/cfg_jump_rd_reg[*]/C \
#Temp #m_fm_loader/loader*/instr_dispatch/cfg_length_reg[*]/C \
#Temp #m_fm_loader/loader*/instr_dispatch/cfg_in_chs_reg[*]/C \
#Temp #m_fm_loader/loader*/instr_dispatch/cfg_pad_idx_reg[*]/C \
#Temp #m_fm_loader/loader*/instr_dispatch/cfg_pad_start_reg[*]/C \
#Temp #m_fm_loader/loader*/instr_dispatch/cfg_pad_end_reg[*]/C \
#Temp #m_fm_loader/loader*/instr_dispatch/cfg_buf_avg_mode_reg[*]/C \
#Temp #m_fm_loader/loader*/instr_dispatch/cfg_buf_wr_bank_id_reg[*]/C"]
#Temp 
#Temp #2019-02-28, add MCP for ELE's initial instruction 
#Temp #FIXME: need Enshan to confirm
#Temp set_multicycle_path -setup 2 -from [get_pins "m_misc_top/ele_top_wrapper/ele_instr_top/reg_jump_read_m1_reg[*][*]/C"]
#Temp set_multicycle_path -hold  1 -from [get_pins "m_misc_top/ele_top_wrapper/ele_instr_top/reg_jump_read_m1_reg[*][*]/C"]
#Temp #set_multicycle_path -setup 2 -from [get_pins "m_misc_top/ele_top_wrapper/ele_instr_top/reg_jump_read_endl_m1_reg[*][*]/C"]
#Temp #set_multicycle_path -hold  1 -from [get_pins "m_misc_top/ele_top_wrapper/ele_instr_top/reg_jump_read_endl_m1_reg[*][*]/C"]
#Temp 
#Temp set_multicycle_path -setup 2 -from [get_pins "m_misc_top/ele_top_wrapper/ele_instr_top/reg_shift_read_reg[*][*]/C"]
#Temp set_multicycle_path -hold  1 -from [get_pins "m_misc_top/ele_top_wrapper/ele_instr_top/reg_shift_read_reg[*][*]/C"]
#Temp 
#Temp set_multicycle_path -setup 2 -from [get_pins "m_misc_top/ele_top_wrapper/ele_instr_top/reg_bank_id_in_reg[*][*]/C"]
#Temp set_multicycle_path -hold  1 -from [get_pins "m_misc_top/ele_top_wrapper/ele_instr_top/reg_bank_id_in_reg[*][*]/C"]
#Temp 
#Temp set_multicycle_path -setup 2 -from [get_pins "m_misc_top/ele_top_wrapper/ele_instr_top/reg_bank_addr_in_reg[*][*]/C"]
#Temp set_multicycle_path -hold  1 -from [get_pins "m_misc_top/ele_top_wrapper/ele_instr_top/reg_bank_addr_in_reg[*][*]/C"]
#Temp 
#Temp #set_multicycle_path -setup 2 -from [get_pins "m_misc_top/ele_top_wrapper/ele_instr_top/reg_jump_write_m1_reg[*][*]/C"]
#Temp #set_multicycle_path -hold  1 -from [get_pins "m_misc_top/ele_top_wrapper/ele_instr_top/reg_jump_write_m1_reg[*][*]/C"]
#Temp #set_multicycle_path -setup 2 -from [get_pins "m_misc_top/ele_top_wrapper/ele_instr_top/reg_jump_write_endl_m1_reg[*][*]/C"]
#Temp #set_multicycle_path -hold  1 -from [get_pins "m_misc_top/ele_top_wrapper/ele_instr_top/reg_jump_write_endl_m1_reg[*][*]/C"]
#Temp 
#Temp #set_multicycle_path -setup 2 -from [get_pins "m_misc_top/ele_top_wrapper/ele_instr_top/reg_shift_write_reg[*][*]/C"]
#Temp #set_multicycle_path -hold  1 -from [get_pins "m_misc_top/ele_top_wrapper/ele_instr_top/reg_shift_write_reg[*][*]/C"]
#Temp 
#Temp #2019-03-09, add MCP for DWC/Conv's relu shifting
#Temp set_multicycle_path -setup 2 -from [get_pins "m_conv_top/inst_pea/GenNLP[*].u_nl/shf_cut_dly_reg[*]/C"] -to [get_pins "m_conv_top/inst_pea/GenNLP[*].u_nl/relu_res_reg[*]/D"]
#Temp set_multicycle_path -hold  1 -from [get_pins "m_conv_top/inst_pea/GenNLP[*].u_nl/shf_cut_dly_reg[*]/C"] -to [get_pins "m_conv_top/inst_pea/GenNLP[*].u_nl/relu_res_reg[*]/D"]
#Temp 
#Temp #FIXME
#Temp #2019-03-13, TEMP XDC need designers review
#Temp #sed -i 's#^.*reg\s\[DATA_BW\s*-1:0\]\s*pipe_reg_nl_relu6_r#(* keep = "true" *) reg \[DATA_BW            -1:0\] pipe_reg_nl_relu6_r#' ../ip/dpu/rtl/cnn/conv/reader/conv_ins_parser.v
#Temp #sed -i 's#^.*reg\s\[NL_TYPE_BW\s*-1:0\]\s*pipe_reg_nl_type_r#(* keep = "true" *) reg \[NL_TYPE_BW         -1:0\] pipe_reg_nl_type_r#' ../ip/dpu/rtl/cnn/conv/reader/conv_ins_parser.v
#Temp set_multicycle_path -setup 2 -from [get_pins " \
#Temp m_conv_top/inst_ins_parser/pipe_reg_nl_relu6_r_reg[*]/C \
#Temp m_conv_top/inst_ins_parser/pipe_reg_nl_type_r_reg[*]/C \
#Temp "]
#Temp set_multicycle_path -hold  1 -from [get_pins " \
#Temp m_conv_top/inst_ins_parser/pipe_reg_nl_relu6_r_reg[*]/C \
#Temp m_conv_top/inst_ins_parser/pipe_reg_nl_type_r_reg[*]/C \
#Temp "]
#Temp 
#Temp #2019-04-01, add false path for ACLK_DR's last-level reset logic
#Temp #set_multicycle_path -setup 2 -from [get_pins " \
#Temp #m_conv_top/inst_pea/GenOcp[*].u_pe/rst_n_rr_reg/C \
#Temp #"]
#Temp #set_multicycle_path -hold  1 -from [get_pins " \
#Temp #m_conv_top/inst_pea/GenOcp[*].u_pe/rst_n_rr_reg/C \
#Temp #"]
#Temp 
#Temp #2019-5-30, add MCP for below path
#Temp #set_multicycle_path -setup 2 -to [get_pins "\
#Temp #m_conv_top/inst_reader/inst_rd_img/u_addr_gen/GenAddr[*].area_w_reg[*]/D \
#Temp #"]
#Temp #set_multicycle_path -hold 1 -to [get_pins "\
#Temp #m_conv_top/inst_reader/inst_rd_img/u_addr_gen/GenAddr[*].area_w_reg[*]/D \
#Temp #"]

##2019-07-25, state-machine will not jump out of IDLE state until start to config plus 3 cycles
##set_multicycle_path -setup 2 -from [get_pins "\
##gen_dpu.m_dpu_*/m_buf_writer/u_ddr_reader/cmd_reg[*]/C \
##"] \
#set_multicycle_path -setup 2 \
#-through [get_nets "\
#gen_dpu.m_dpu_*/m_buf_writer/u_ddr_reader/mul_vld_byte_hp0[*] \
#gen_dpu.m_dpu_*/m_buf_writer/u_ddr_reader/mul_vld_byte_hp2[*] \
#"]
#
##set_multicycle_path -hold  1 -from [get_pins "\
##gen_dpu.m_dpu_*/m_buf_writer/u_ddr_reader/cmd_reg[*]/C \
##"] \
#set_multicycle_path -hold  1 \
#-through [get_nets "\
#gen_dpu.m_dpu_*/m_buf_writer/u_ddr_reader/mul_vld_byte_hp0[*] \
#gen_dpu.m_dpu_*/m_buf_writer/u_ddr_reader/mul_vld_byte_hp2[*] \
#"]
#
#set_multicycle_path -setup 2 -from [get_pins "\
#m_wgt_load_*/u_ddr_reader/cmd_reg[*]/C \
#"] \
#-through [get_nets "\
#m_wgt_load_*/u_ddr_reader/mul_vld_byte_hp0[*] \
#m_wgt_load_*/u_ddr_reader/mul_vld_byte_hp2[*] \
#"]
#
#set_multicycle_path -hold  1 -from [get_pins "\
#m_wgt_load_*/u_ddr_reader/cmd_reg[*]/C \
#"] \
#-through [get_nets "\
#m_wgt_load_*/u_ddr_reader/mul_vld_byte_hp0[*] \
#m_wgt_load_*/u_ddr_reader/mul_vld_byte_hp2[*] \
#"]
set_multicycle_path -setup 2 -to [get_pins "\
gen_dpu.m_dpu_*/m_img_load_top/gen_loader_port[*].gen_loader_aug[*].loader/ddr_reader/total_nbyte_rd_reg/* \
gen_dpu.m_dpu_*/m_img_load_top/gen_loader_port[*].gen_loader_aug[*].loader/ddr_reader/total_nbyte_wr_reg/* \
m_wgt_load_*/gen_loader_port[*].gen_loader_aug[*].loader/ddr_reader/total_nbyte_rd_reg/* \
m_wgt_load_*/gen_loader_port[*].gen_loader_aug[*].loader/ddr_reader/total_nbyte_wr_reg/* \
"]
set_multicycle_path -hold 1 -to [get_pins "\
gen_dpu.m_dpu_*/m_img_load_top/gen_loader_port[*].gen_loader_aug[*].loader/ddr_reader/total_nbyte_rd_reg/* \
gen_dpu.m_dpu_*/m_img_load_top/gen_loader_port[*].gen_loader_aug[*].loader/ddr_reader/total_nbyte_wr_reg/* \
m_wgt_load_*/gen_loader_port[*].gen_loader_aug[*].loader/ddr_reader/total_nbyte_rd_reg/* \
m_wgt_load_*/gen_loader_port[*].gen_loader_aug[*].loader/ddr_reader/total_nbyte_wr_reg/* \
"]

#2019-10-25, add MCP for clock gating
set_multicycle_path -setup 2 -from [get_pins "\
m_scheduler_top/m_scheduler_thdl/m_lctrl/pre_cmd_reg[1]/C \
"] \
-to [get_pins "\
gen_clk.m_bufgce_lw/CE \
"]
set_multicycle_path -hold  1 -from [get_pins "\
m_scheduler_top/m_scheduler_thdl/m_lctrl/pre_cmd_reg[1]/C \
"] \
-to [get_pins "\
gen_clk.m_bufgce_lw/CE \
"]

set_multicycle_path -setup 2 -from [get_pins "\
m_scheduler_top/m_scheduler_thdl/m_lctrl/pre_cmd_reg[1]/C \
m_scheduler_top/m_scheduler_thdl/m_cctrl/pre_cmd_reg/C \
"] \
-to [get_pins "\
gen_clk.m_bufgce_sw/CE \
"]
set_multicycle_path -hold  1 -from [get_pins "\
m_scheduler_top/m_scheduler_thdl/m_lctrl/pre_cmd_reg[1]/C \
m_scheduler_top/m_scheduler_thdl/m_cctrl/pre_cmd_reg/C \
"] \
-to [get_pins "\
gen_clk.m_bufgce_sw/CE \
"]

set_multicycle_path -setup 2 -from [get_pins "\
m_scheduler_top/m_scheduler_thdl/m_lctrl/pre_cmd_reg[0]/C\
"] \
-to [get_pins "\
gen_clk.m_bufgce_li/CE \
"]
set_multicycle_path -hold  1 -from [get_pins "\
m_scheduler_top/m_scheduler_thdl/m_lctrl/pre_cmd_reg[0]/C\
"] \
-to [get_pins "\
gen_clk.m_bufgce_li/CE \
"]

set_multicycle_path -setup 2 -from [get_pins "\
m_scheduler_top/m_scheduler_thdl/m_sctrl/pre_cmd_reg/C\
"] \
-to [get_pins "\
gen_clk.m_bufgce_s/CE \
"]
set_multicycle_path -hold  1 -from [get_pins "\
m_scheduler_top/m_scheduler_thdl/m_sctrl/pre_cmd_reg/C\
"] \
-to [get_pins "\
gen_clk.m_bufgce_s/CE \
"]

set_multicycle_path -setup 2 -from [get_pins "\
m_scheduler_top/m_scheduler_thdl/m_cctrl/pre_cmd_reg/C\
"] \
-to [get_pins "\
gen_clk.m_bufgce_c/CE \
"]
set_multicycle_path -hold  1 -from [get_pins "\
m_scheduler_top/m_scheduler_thdl/m_cctrl/pre_cmd_reg/C\
"] \
-to [get_pins "\
gen_clk.m_bufgce_c/CE \
"]

set_multicycle_path -setup 2 -from [get_pins "\
m_scheduler_top/m_scheduler_thdl/m_mctrl/pre_cmd_reg/C\
"] \
-to [get_pins "\
gen_clk.m_bufgce_m/CE \
"]
set_multicycle_path -hold  1 -from [get_pins "\
m_scheduler_top/m_scheduler_thdl/m_mctrl/pre_cmd_reg/C\
"] \
-to [get_pins "\
gen_clk.m_bufgce_m/CE \
"]

set_multicycle_path -setup 2 -from [get_pins "\
m_scheduler_top/m_scheduler_thdl/m_lctrl/pre_cmd_reg[0]/C\
m_scheduler_top/m_scheduler_thdl/m_sctrl/pre_cmd_reg/C\
m_scheduler_top/m_scheduler_thdl/m_cctrl/pre_cmd_reg/C\
m_scheduler_top/m_scheduler_thdl/m_mctrl/pre_cmd_reg/C\
"] \
-to [get_pins "\
gen_clk.m_bufgce_b/CE \
"]
set_multicycle_path -hold  1 -from [get_pins "\
m_scheduler_top/m_scheduler_thdl/m_lctrl/pre_cmd_reg[0]/C\
m_scheduler_top/m_scheduler_thdl/m_sctrl/pre_cmd_reg/C\
m_scheduler_top/m_scheduler_thdl/m_cctrl/pre_cmd_reg/C\
m_scheduler_top/m_scheduler_thdl/m_mctrl/pre_cmd_reg/C\
"] \
-to [get_pins "\
gen_clk.m_bufgce_b/CE \
"]

#2019-12-18, for conv's new clock gating polocy
#set_multicycle_path -setup 4 -from [get_pins "\
#m_scheduler_top/m_scheduler_thdl/m_cctrl/pre_cmd_reg/C\
#"] \
#-to [get_pins "\
#gen_clk.m_bufgce_c_dr/CE \
#"]
#set_multicycle_path -hold  3 -from [get_pins "\
#m_scheduler_top/m_scheduler_thdl/m_cctrl/pre_cmd_reg/C\
#"] \
#-to [get_pins "\
#gen_clk.m_bufgce_c_dr/CE \
#"]
set_multicycle_path -setup 4 -to [get_pins "\
gen_clk.m_bufgce_c_dr/CE \
"]
set_multicycle_path -hold  3 -to [get_pins "\
gen_clk.m_bufgce_c_dr/CE \
"]
set_multicycle_path -setup 2 -to [get_pins "\
gen_clk.m_bufgce_cs/CE \
"]
set_multicycle_path -hold  1 -to [get_pins "\
gen_clk.m_bufgce_cs/CE \
"]

#2020-01-16, add reset's XDC in dpu_clock_gen
set_multicycle_path -setup 2 -from [get_pins " \ 
dpu_clock_gen_inst/RESET_REG_CONTROL[3].reset_reg_counter_reg[3]*/C\
"]
set_multicycle_path -hold  1 -from [get_pins " \ 
dpu_clock_gen_inst/RESET_REG_CONTROL[3].reset_reg_counter_reg[3]*/C\
"]
