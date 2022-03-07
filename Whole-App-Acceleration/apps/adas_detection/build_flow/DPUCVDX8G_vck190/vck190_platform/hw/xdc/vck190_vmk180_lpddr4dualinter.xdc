
###############################################################################
#
#     channel 0 LP0
#
###############################################################################

#create_clock -period 5.000 -name sys_clk0_0_clk_p [get_ports sys_clk0_0_clk_p]
#set_clock_uncertainty -hold 0.200 [get_clocks clkout1_primitive]


set_property PACKAGE_PIN AK8 [get_ports sys_clk0_0_clk_p]
set_property PACKAGE_PIN AK7 [get_ports sys_clk0_0_clk_n]
#set_property IOSTANDARD LVDS15 [get_ports sys_clk0_0_clk_p]
#set_property IOSTANDARD LVDS15 [get_ports sys_clk0_0_clk_n]
set_property IOSTANDARD DIFF_LVSTL_11 [get_ports sys_clk0_0_clk_p]
set_property IOSTANDARD DIFF_LVSTL_11 [get_ports sys_clk0_0_clk_n]

#set_property PACKAGE_PIN AE42  [get_ports SYS_CLK1_IN_0_clk_p]
#set_property PACKAGE_PIN AF43  [get_ports SYS_CLK1_IN_0_clk_n]
#set_property IOSTANDARD DIFF_LVSTL_11 [get_ports SYS_CLK1_IN_0_clk_p]
#set_property IOSTANDARD DIFF_LVSTL_11 [get_ports SYS_CLK1_IN_0_clk_n]


#set_property LOC NOC_NMU512_X0Y0 [get_cells {design_1_i/axi_noc_0/inst/S00_AXI_nmu/*_nmu_0_top_INST/NOC_NMU512_INST}]

#set_false_path -from [get_clocks clk_pl_0] -to [get_clocks clkout1_primitive] 
#set_false_path -from [get_clocks clkout1_primitive] -to [get_clocks clk_pl_0]




set_property	PACKAGE_PIN	AL3	[get_ports "CH0_LPDDR4_0_0_ca_a[0]"] 	
set_property	PACKAGE_PIN	AM4	[get_ports "CH0_LPDDR4_0_0_ca_a[1]"] 	
set_property	PACKAGE_PIN	AK2	[get_ports "CH0_LPDDR4_0_0_ca_a[2]"] 	
set_property	PACKAGE_PIN	AH1	[get_ports "CH0_LPDDR4_0_0_ca_a[3]"] 	
set_property	PACKAGE_PIN	AG4	[get_ports "CH0_LPDDR4_0_0_ca_a[4]"] 	
set_property	PACKAGE_PIN	AH3	[get_ports "CH0_LPDDR4_0_0_ca_a[5]"] 	
set_property	PACKAGE_PIN	AM6	[get_ports "CH0_LPDDR4_0_0_ca_b[0]"] 	
set_property	PACKAGE_PIN	AL4	[get_ports "CH0_LPDDR4_0_0_ca_b[1]"] 	
set_property	PACKAGE_PIN	AH5	[get_ports "CH0_LPDDR4_0_0_ca_b[2]"] 	
set_property	PACKAGE_PIN	AJ5	[get_ports "CH0_LPDDR4_0_0_ca_b[3]"] 	
set_property	PACKAGE_PIN	AH6	[get_ports "CH0_LPDDR4_0_0_ca_b[4]"] 	
set_property	PACKAGE_PIN	AH7	[get_ports "CH0_LPDDR4_0_0_ca_b[5]"] 	
set_property	PACKAGE_PIN	AJ3	[get_ports "CH0_LPDDR4_0_0_ck_c_a[0]"] 	
set_property	PACKAGE_PIN	AK4	[get_ports "CH0_LPDDR4_0_0_ck_c_b[0]"] 	
set_property	PACKAGE_PIN	AJ2	[get_ports "CH0_LPDDR4_0_0_ck_t_a[0]"] 	
set_property	PACKAGE_PIN	AJ4	[get_ports "CH0_LPDDR4_0_0_ck_t_b[0]"] 	
set_property	PACKAGE_PIN	AL2	[get_ports "CH0_LPDDR4_0_0_cke_a[0]"] 	
set_property	PACKAGE_PIN	AM5	[get_ports "CH0_LPDDR4_0_0_cke_b[0]"] 	
set_property	PACKAGE_PIN	AH2	[get_ports "CH0_LPDDR4_0_0_cs_a[0]"] 	
set_property	PACKAGE_PIN	AK13	[get_ports "CH0_LPDDR4_0_0_cs_b[0]"] 	
set_property	PACKAGE_PIN	AN3	[get_ports "CH0_LPDDR4_0_0_dmi_a[0]"] 	
set_property	PACKAGE_PIN	AL11	[get_ports "CH0_LPDDR4_0_0_dmi_a[1]"] 	
set_property	PACKAGE_PIN	AT8	[get_ports "CH0_LPDDR4_0_0_dmi_b[0]"] 	
set_property	PACKAGE_PIN	AJ12	[get_ports "CH0_LPDDR4_0_0_dmi_b[1]"] 	
set_property	PACKAGE_PIN	AN1	[get_ports "CH0_LPDDR4_0_0_dq_a[0]"] 
set_property	PACKAGE_PIN	AP2	[get_ports "CH0_LPDDR4_0_0_dq_a[1]"] 
set_property	PACKAGE_PIN	AH9	[get_ports "CH0_LPDDR4_0_0_dq_a[10]"]
set_property	PACKAGE_PIN	AJ10	[get_ports "CH0_LPDDR4_0_0_dq_a[11]"]
set_property	PACKAGE_PIN	AM12	[get_ports "CH0_LPDDR4_0_0_dq_a[12]"]
set_property	PACKAGE_PIN	AM10	[get_ports "CH0_LPDDR4_0_0_dq_a[13]"]
set_property	PACKAGE_PIN	AL12	[get_ports "CH0_LPDDR4_0_0_dq_a[14]"]
set_property	PACKAGE_PIN	AM11	[get_ports "CH0_LPDDR4_0_0_dq_a[15]"]
set_property	PACKAGE_PIN	AR4	[get_ports "CH0_LPDDR4_0_0_dq_a[2]"] 
set_property	PACKAGE_PIN	AT4	[get_ports "CH0_LPDDR4_0_0_dq_a[3]"] 
set_property	PACKAGE_PIN	AP4	[get_ports "CH0_LPDDR4_0_0_dq_a[4]"] 
set_property	PACKAGE_PIN	AP3	[get_ports "CH0_LPDDR4_0_0_dq_a[5]"] 
set_property	PACKAGE_PIN	AN2	[get_ports "CH0_LPDDR4_0_0_dq_a[6]"] 
set_property	PACKAGE_PIN	AM1	[get_ports "CH0_LPDDR4_0_0_dq_a[7]"] 
set_property	PACKAGE_PIN	AH11	[get_ports "CH0_LPDDR4_0_0_dq_a[8]"] 
set_property	PACKAGE_PIN	AH10	[get_ports "CH0_LPDDR4_0_0_dq_a[9]"] 
set_property	PACKAGE_PIN	AT9	[get_ports "CH0_LPDDR4_0_0_dq_b[0]"] 
set_property	PACKAGE_PIN	AN10	[get_ports "CH0_LPDDR4_0_0_dq_b[1]"] 
set_property	PACKAGE_PIN	AH13	[get_ports "CH0_LPDDR4_0_0_dq_b[10]"]
set_property	PACKAGE_PIN	AH12	[get_ports "CH0_LPDDR4_0_0_dq_b[11]"]
set_property	PACKAGE_PIN	AM15	[get_ports "CH0_LPDDR4_0_0_dq_b[12]"]
set_property	PACKAGE_PIN	AM14	[get_ports "CH0_LPDDR4_0_0_dq_b[13]"]
set_property	PACKAGE_PIN	AL15	[get_ports "CH0_LPDDR4_0_0_dq_b[14]"]
set_property	PACKAGE_PIN	AL13	[get_ports "CH0_LPDDR4_0_0_dq_b[15]"]
set_property	PACKAGE_PIN	AP10	[get_ports "CH0_LPDDR4_0_0_dq_b[2]"] 
set_property	PACKAGE_PIN	AT10	[get_ports "CH0_LPDDR4_0_0_dq_b[3]"] 
set_property	PACKAGE_PIN	AN8	[get_ports "CH0_LPDDR4_0_0_dq_b[4]"] 
set_property	PACKAGE_PIN	AP7	[get_ports "CH0_LPDDR4_0_0_dq_b[5]"] 
set_property	PACKAGE_PIN	AP8	[get_ports "CH0_LPDDR4_0_0_dq_b[6]"] 
set_property	PACKAGE_PIN	AR7	[get_ports "CH0_LPDDR4_0_0_dq_b[7]"] 
set_property	PACKAGE_PIN	AH15	[get_ports "CH0_LPDDR4_0_0_dq_b[8]"] 
set_property	PACKAGE_PIN	AJ15	[get_ports "CH0_LPDDR4_0_0_dq_b[9]"] 

set_property	PACKAGE_PIN	AR1	[get_ports "CH0_LPDDR4_0_0_dqs_c_a[0]"] 
set_property	PACKAGE_PIN	AR2	[get_ports "CH0_LPDDR4_0_0_dqs_t_a[0]"] 

set_property	PACKAGE_PIN	AK9	[get_ports "CH0_LPDDR4_0_0_dqs_c_a[1]"]	
set_property	PACKAGE_PIN	AK10	[get_ports "CH0_LPDDR4_0_0_dqs_t_a[1]"] 

set_property	PACKAGE_PIN	AR9	[get_ports "CH0_LPDDR4_0_0_dqs_c_b[0]"] 	
set_property	PACKAGE_PIN	AP9	[get_ports "CH0_LPDDR4_0_0_dqs_t_b[0]"]
 	
set_property	PACKAGE_PIN	AJ14	[get_ports "CH0_LPDDR4_0_0_dqs_c_b[1]"] 
set_property	PACKAGE_PIN	AK14	[get_ports "CH0_LPDDR4_0_0_dqs_t_b[1]"] 
	
set_property	PACKAGE_PIN	AM7	[get_ports "CH0_LPDDR4_0_0_reset_n[0]"] 

###############################################################################
#
#     channel 1 LP0
#
###############################################################################

set_property	PACKAGE_PIN	BG8	[get_ports "CH1_LPDDR4_0_0_ca_a[0]"] 	
set_property	PACKAGE_PIN	BG9	[get_ports "CH1_LPDDR4_0_0_ca_a[1]"] 	
set_property	PACKAGE_PIN	BB9	[get_ports "CH1_LPDDR4_0_0_ca_a[2]"] 	
set_property	PACKAGE_PIN	BC10	[get_ports "CH1_LPDDR4_0_0_ca_a[3]"] 	
set_property	PACKAGE_PIN	BB10	[get_ports "CH1_LPDDR4_0_0_ca_a[4]"] 	
set_property	PACKAGE_PIN	BA9	[get_ports "CH1_LPDDR4_0_0_ca_a[5]"] 	
set_property	PACKAGE_PIN	BF4	[get_ports "CH1_LPDDR4_0_0_ca_b[0]"] 	
set_property	PACKAGE_PIN	BE4	[get_ports "CH1_LPDDR4_0_0_ca_b[1]"] 	
set_property	PACKAGE_PIN	BA4	[get_ports "CH1_LPDDR4_0_0_ca_b[2]"] 	
set_property	PACKAGE_PIN	AY4	[get_ports "CH1_LPDDR4_0_0_ca_b[3]"] 	
set_property	PACKAGE_PIN	AY5	[get_ports "CH1_LPDDR4_0_0_ca_b[4]"] 	
set_property	PACKAGE_PIN	BB5	[get_ports "CH1_LPDDR4_0_0_ca_b[5]"] 	
set_property	PACKAGE_PIN	BD9	[get_ports "CH1_LPDDR4_0_0_ck_c_a[0]"] 	
set_property	PACKAGE_PIN	BB4	[get_ports "CH1_LPDDR4_0_0_ck_c_b[0]"] 	
set_property	PACKAGE_PIN	BD10	[get_ports "CH1_LPDDR4_0_0_ck_t_a[0]"] 	
set_property	PACKAGE_PIN	BB3	[get_ports "CH1_LPDDR4_0_0_ck_t_b[0]"] 	
set_property	PACKAGE_PIN	BG10	[get_ports "CH1_LPDDR4_0_0_cke_a[0]"] 	
set_property	PACKAGE_PIN	BC3	[get_ports "CH1_LPDDR4_0_0_cke_b[0]"] 	
set_property	PACKAGE_PIN	BE9	[get_ports "CH1_LPDDR4_0_0_cs_a[0]"] 	
set_property	PACKAGE_PIN	BC7	[get_ports "CH1_LPDDR4_0_0_cs_b[0]"] 	
set_property	PACKAGE_PIN	BC8	[get_ports "CH1_LPDDR4_0_0_dmi_a[0]"] 	
set_property	PACKAGE_PIN	AT3	[get_ports "CH1_LPDDR4_0_0_dmi_a[1]"] 	
set_property	PACKAGE_PIN	AV9	[get_ports "CH1_LPDDR4_0_0_dmi_b[0]"] 	
set_property	PACKAGE_PIN	BC2	[get_ports "CH1_LPDDR4_0_0_dmi_b[1]"] 	
set_property	PACKAGE_PIN	BD8	[get_ports "CH1_LPDDR4_0_0_dq_a[0]"] 
set_property	PACKAGE_PIN	BF8	[get_ports "CH1_LPDDR4_0_0_dq_a[1]"] 
set_property	PACKAGE_PIN	AV1	[get_ports "CH1_LPDDR4_0_0_dq_a[10]"]
set_property	PACKAGE_PIN	AU1	[get_ports "CH1_LPDDR4_0_0_dq_a[11]"]
set_property	PACKAGE_PIN	AW5	[get_ports "CH1_LPDDR4_0_0_dq_a[12]"]
set_property	PACKAGE_PIN	AW4	[get_ports "CH1_LPDDR4_0_0_dq_a[13]"]
set_property	PACKAGE_PIN	AV3	[get_ports "CH1_LPDDR4_0_0_dq_a[14]"]
set_property	PACKAGE_PIN	AW3	[get_ports "CH1_LPDDR4_0_0_dq_a[15]"]
set_property	PACKAGE_PIN	BF7	[get_ports "CH1_LPDDR4_0_0_dq_a[2]"] 
set_property	PACKAGE_PIN	BE7	[get_ports "CH1_LPDDR4_0_0_dq_a[3]"] 
set_property	PACKAGE_PIN	AY6	[get_ports "CH1_LPDDR4_0_0_dq_a[4]"] 
set_property	PACKAGE_PIN	AY7	[get_ports "CH1_LPDDR4_0_0_dq_a[5]"] 
set_property	PACKAGE_PIN	BB8	[get_ports "CH1_LPDDR4_0_0_dq_a[6]"] 
set_property	PACKAGE_PIN	BA8	[get_ports "CH1_LPDDR4_0_0_dq_a[7]"] 
set_property	PACKAGE_PIN	AW2	[get_ports "CH1_LPDDR4_0_0_dq_a[8]"] 
set_property	PACKAGE_PIN	AV2	[get_ports "CH1_LPDDR4_0_0_dq_a[9]"] 
set_property	PACKAGE_PIN	AW9	[get_ports "CH1_LPDDR4_0_0_dq_b[0]"] 
set_property	PACKAGE_PIN	AY9	[get_ports "CH1_LPDDR4_0_0_dq_b[1]"] 
set_property	PACKAGE_PIN	BA2	[get_ports "CH1_LPDDR4_0_0_dq_b[10]"]
set_property	PACKAGE_PIN	AY1	[get_ports "CH1_LPDDR4_0_0_dq_b[11]"]
set_property	PACKAGE_PIN	BF3	[get_ports "CH1_LPDDR4_0_0_dq_b[12]"]
set_property	PACKAGE_PIN	BD3	[get_ports "CH1_LPDDR4_0_0_dq_b[13]"]
set_property	PACKAGE_PIN	BD2	[get_ports "CH1_LPDDR4_0_0_dq_b[14]"]
set_property	PACKAGE_PIN	BE2	[get_ports "CH1_LPDDR4_0_0_dq_b[15]"]
set_property	PACKAGE_PIN	AV10	[get_ports "CH1_LPDDR4_0_0_dq_b[2]"] 
set_property	PACKAGE_PIN	AY10	[get_ports "CH1_LPDDR4_0_0_dq_b[3]"] 
set_property	PACKAGE_PIN	AU6	[get_ports "CH1_LPDDR4_0_0_dq_b[4]"] 
set_property	PACKAGE_PIN	AV6	[get_ports "CH1_LPDDR4_0_0_dq_b[5]"] 
set_property	PACKAGE_PIN	AW7	[get_ports "CH1_LPDDR4_0_0_dq_b[6]"] 
set_property	PACKAGE_PIN	AV7	[get_ports "CH1_LPDDR4_0_0_dq_b[7]"] 
set_property	PACKAGE_PIN	AY2	[get_ports "CH1_LPDDR4_0_0_dq_b[8]"] 
set_property	PACKAGE_PIN	BA3	[get_ports "CH1_LPDDR4_0_0_dq_b[9]"]  

set_property	PACKAGE_PIN	BC6	[get_ports "CH1_LPDDR4_0_0_dqs_c_a[0]"] 
set_property	PACKAGE_PIN	BD7	[get_ports "CH1_LPDDR4_0_0_dqs_t_a[0]"] 

set_property	PACKAGE_PIN	AT1	[get_ports "CH1_LPDDR4_0_0_dqs_c_a[1]"] 
set_property	PACKAGE_PIN	AT2	[get_ports "CH1_LPDDR4_0_0_dqs_t_a[1]"] 
	
set_property	PACKAGE_PIN	AU9	[get_ports "CH1_LPDDR4_0_0_dqs_c_b[0]"] 
set_property	PACKAGE_PIN	AU8	[get_ports "CH1_LPDDR4_0_0_dqs_t_b[0]"] 

set_property	PACKAGE_PIN	BB1	[get_ports "CH1_LPDDR4_0_0_dqs_c_b[1]"] 
set_property	PACKAGE_PIN	BA1	[get_ports "CH1_LPDDR4_0_0_dqs_t_b[1]"] 
		
	
set_property	PACKAGE_PIN	AL7	[get_ports "CH1_LPDDR4_0_0_reset_n[0]"] 	
	


###############################################################################
#
#     channel 0 LP1
#
###############################################################################

#create_clock -period 5.000 -name sys_clk1_0_clk_p [get_ports sys_clk1_0_clk_p]
#set_clock_uncertainty -hold 0.200 [get_clocks clkout1_primitive]


set_property	PACKAGE_PIN	AW27 	[get_ports sys_clk1_0_clk_p] 
set_property	PACKAGE_PIN	AY27	[get_ports sys_clk1_0_clk_n] 
set_property IOSTANDARD DIFF_LVSTL_11   [get_ports sys_clk1_0_clk_p]
set_property IOSTANDARD DIFF_LVSTL_11   [get_ports sys_clk1_0_clk_n]

#set_property PACKAGE_PIN AE42  [get_ports SYS_CLK1_IN_0_clk_p]
#set_property PACKAGE_PIN AF43  [get_ports SYS_CLK1_IN_0_clk_n]
#set_property IOSTANDARD DIFF_LVSTL_11 [get_ports SYS_CLK1_IN_0_clk_p]
#set_property IOSTANDARD DIFF_LVSTL_11 [get_ports SYS_CLK1_IN_0_clk_n]


#set_property LOC NOC_NMU512_X0Y0 [get_cells {design_1_i/axi_noc_0/inst/S00_AXI_nmu/*_nmu_0_top_INST/NOC_NMU512_INST}]

#set_false_path -from [get_clocks clk_pl_0] -to [get_clocks clkout1_primitive] 
#set_false_path -from [get_clocks clkout1_primitive] -to [get_clocks clk_pl_0]




set_property	PACKAGE_PIN	BE29	[get_ports "CH0_LPDDR4_1_0_ca_a[0]"] ;;
set_property	PACKAGE_PIN	BE30	[get_ports "CH0_LPDDR4_1_0_ca_a[1]"] ;
set_property	PACKAGE_PIN	BF29	[get_ports "CH0_LPDDR4_1_0_ca_a[2]"] ;
set_property	PACKAGE_PIN	BF27	[get_ports "CH0_LPDDR4_1_0_ca_a[3]"];
set_property	PACKAGE_PIN	BF26	[get_ports "CH0_LPDDR4_1_0_ca_a[4]"] ;
set_property	PACKAGE_PIN	BG26	[get_ports "CH0_LPDDR4_1_0_ca_a[5]"] ;
set_property	PACKAGE_PIN	BB29	[get_ports "CH0_LPDDR4_1_0_ca_b[0]"] ;
set_property	PACKAGE_PIN	BD29	[get_ports "CH0_LPDDR4_1_0_ca_b[1]"] ;
set_property	PACKAGE_PIN	BC26	[get_ports "CH0_LPDDR4_1_0_ca_b[2]"] ;
set_property	PACKAGE_PIN	BB26	[get_ports "CH0_LPDDR4_1_0_ca_b[3]"] ;
set_property	PACKAGE_PIN	BA27	[get_ports "CH0_LPDDR4_1_0_ca_b[4]"] ;
set_property	PACKAGE_PIN	BA28	[get_ports "CH0_LPDDR4_1_0_ca_b[5]"] ;
set_property	PACKAGE_PIN	BF28	[get_ports "CH0_LPDDR4_1_0_ck_c_a[0]"] ;
set_property	PACKAGE_PIN	BC27	[get_ports "CH0_LPDDR4_1_0_ck_c_b[0]"] ;
set_property	PACKAGE_PIN	BE27	[get_ports "CH0_LPDDR4_1_0_ck_t_a[0]"] ;
set_property	PACKAGE_PIN	BD27	[get_ports "CH0_LPDDR4_1_0_ck_t_b[0]"] ;
set_property	PACKAGE_PIN	BG30	[get_ports "CH0_LPDDR4_1_0_cke_a[0]"] ;
set_property	PACKAGE_PIN	BD30	[get_ports "CH0_LPDDR4_1_0_cke_b[0]"] ;
set_property	PACKAGE_PIN	BE26	[get_ports "CH0_LPDDR4_1_0_cs_a[0]"] ;
set_property	PACKAGE_PIN	AP27	[get_ports "CH0_LPDDR4_1_0_cs_b[0]"] ;
set_property	PACKAGE_PIN	BC31	[get_ports "CH0_LPDDR4_1_0_dmi_a[0]"] ;
set_property	PACKAGE_PIN	AU28	[get_ports "CH0_LPDDR4_1_0_dmi_a[1]"] ;
set_property	PACKAGE_PIN	AN32	[get_ports "CH0_LPDDR4_1_0_dmi_b[0]"] ;
set_property	PACKAGE_PIN	AR26	[get_ports "CH0_LPDDR4_1_0_dmi_b[1]"] ;
set_property	PACKAGE_PIN	BG31	[get_ports "CH0_LPDDR4_1_0_dq_a[0]"] ;
set_property	PACKAGE_PIN	BF31	[get_ports "CH0_LPDDR4_1_0_dq_a[1]"] ;
set_property	PACKAGE_PIN	AT26	[get_ports "CH0_LPDDR4_1_0_dq_a[10]"] ;
set_property	PACKAGE_PIN	AT25	[get_ports "CH0_LPDDR4_1_0_dq_a[11]"] ;
set_property	PACKAGE_PIN	AT29	[get_ports "CH0_LPDDR4_1_0_dq_a[12]"] ;
set_property	PACKAGE_PIN	AR29	[get_ports "CH0_LPDDR4_1_0_dq_a[13]"] ;
set_property	PACKAGE_PIN	AU29	[get_ports "CH0_LPDDR4_1_0_dq_a[14]"] ;
set_property	PACKAGE_PIN	AT28	[get_ports "CH0_LPDDR4_1_0_dq_a[15]"] ;
set_property	PACKAGE_PIN	BE32	[get_ports "CH0_LPDDR4_1_0_dq_a[2]"] ;
set_property	PACKAGE_PIN	BF32	[get_ports "CH0_LPDDR4_1_0_dq_a[3]"] ;
set_property	PACKAGE_PIN	BB33	[get_ports "CH0_LPDDR4_1_0_dq_a[4]"] ;
set_property	PACKAGE_PIN	BB30	[get_ports "CH0_LPDDR4_1_0_dq_a[5]"] ;
set_property	PACKAGE_PIN	BC32	[get_ports "CH0_LPDDR4_1_0_dq_a[6]"] ;
set_property	PACKAGE_PIN	BB31	[get_ports "CH0_LPDDR4_1_0_dq_a[7]"] ;
set_property	PACKAGE_PIN	AU25	[get_ports "CH0_LPDDR4_1_0_dq_a[8]"] ;
set_property	PACKAGE_PIN	AV25	[get_ports "CH0_LPDDR4_1_0_dq_a[9]"] ;
set_property	PACKAGE_PIN	AT31	[get_ports "CH0_LPDDR4_1_0_dq_b[0]"] ;
set_property	PACKAGE_PIN	AV30	[get_ports "CH0_LPDDR4_1_0_dq_b[1]"] ;
set_property	PACKAGE_PIN	AM27	[get_ports "CH0_LPDDR4_1_0_dq_b[10]"] ;
set_property	PACKAGE_PIN	AM26	[get_ports "CH0_LPDDR4_1_0_dq_b[11]"] ;
set_property	PACKAGE_PIN	AN29	[get_ports "CH0_LPDDR4_1_0_dq_b[12]"] ;
set_property	PACKAGE_PIN	AM29	[get_ports "CH0_LPDDR4_1_0_dq_b[13]"] ;
set_property	PACKAGE_PIN	AN26	[get_ports "CH0_LPDDR4_1_0_dq_b[14]"] ;
set_property	PACKAGE_PIN	AN28	[get_ports "CH0_LPDDR4_1_0_dq_b[15]"] ;
set_property	PACKAGE_PIN	AU31	[get_ports "CH0_LPDDR4_1_0_dq_b[2]"] ;
set_property	PACKAGE_PIN	AV31	[get_ports "CH0_LPDDR4_1_0_dq_b[3]"] ;
set_property	PACKAGE_PIN	AM30	[get_ports "CH0_LPDDR4_1_0_dq_b[4]"] ;
set_property	PACKAGE_PIN	AN31	[get_ports "CH0_LPDDR4_1_0_dq_b[5]"] ;
set_property	PACKAGE_PIN	AM33	[get_ports "CH0_LPDDR4_1_0_dq_b[6]"] ;
set_property	PACKAGE_PIN	AM32	[get_ports "CH0_LPDDR4_1_0_dq_b[7]"] ;
set_property	PACKAGE_PIN	AP25	[get_ports "CH0_LPDDR4_1_0_dq_b[8]"] ;
set_property	PACKAGE_PIN	AN25	[get_ports "CH0_LPDDR4_1_0_dq_b[9]"] ;

set_property	PACKAGE_PIN	BD32	[get_ports "CH0_LPDDR4_1_0_dqs_c_a[0]"] ;
set_property	PACKAGE_PIN	BE31	[get_ports "CH0_LPDDR4_1_0_dqs_t_a[0]"] ;

set_property	PACKAGE_PIN	AU27	[get_ports "CH0_LPDDR4_1_0_dqs_c_a[1]"] ;
set_property	PACKAGE_PIN	AV26	[get_ports "CH0_LPDDR4_1_0_dqs_t_a[1]"] ;

set_property	PACKAGE_PIN	AP30	[get_ports "CH0_LPDDR4_1_0_dqs_c_b[0]"] ;
set_property	PACKAGE_PIN	AR30	[get_ports "CH0_LPDDR4_1_0_dqs_t_b[0]"] ;

set_property	PACKAGE_PIN	AR27	[get_ports "CH0_LPDDR4_1_0_dqs_c_b[1]"] ;
set_property	PACKAGE_PIN	AP28	[get_ports "CH0_LPDDR4_1_0_dqs_t_b[1]"] ;


set_property	PACKAGE_PIN	AV29	[get_ports "CH0_LPDDR4_1_0_reset_n[0]"] ;

###############################################################################
#
#     channel 1 LP1
#
###############################################################################

set_property	PACKAGE_PIN	AM35	[get_ports "CH1_LPDDR4_1_0_ca_a[0]"] ;
set_property	PACKAGE_PIN	AM36	[get_ports "CH1_LPDDR4_1_0_ca_a[1]"] ;
set_property	PACKAGE_PIN	AT35	[get_ports "CH1_LPDDR4_1_0_ca_a[2]"] ;
set_property	PACKAGE_PIN	AR35	[get_ports "CH1_LPDDR4_1_0_ca_a[3]"] ;
set_property	PACKAGE_PIN	AN35	[get_ports "CH1_LPDDR4_1_0_ca_a[4]"] ;
set_property	PACKAGE_PIN	AP36	[get_ports "CH1_LPDDR4_1_0_ca_a[5]"] ;
set_property	PACKAGE_PIN	BB38	[get_ports "CH1_LPDDR4_1_0_ca_b[0]"] ;
set_property	PACKAGE_PIN	BC38	[get_ports "CH1_LPDDR4_1_0_ca_b[1]"] ;
set_property	PACKAGE_PIN	BC36	[get_ports "CH1_LPDDR4_1_0_ca_b[2]"] ;
set_property	PACKAGE_PIN	BC35	[get_ports "CH1_LPDDR4_1_0_ca_b[3]"] ;
set_property	PACKAGE_PIN	BB35	[get_ports "CH1_LPDDR4_1_0_ca_b[4]"] ;
set_property	PACKAGE_PIN	BB36	[get_ports "CH1_LPDDR4_1_0_ca_b[5]"] ;
set_property	PACKAGE_PIN	AP37	[get_ports "CH1_LPDDR4_1_0_ck_c_a[0]"] ;
set_property	PACKAGE_PIN	BC37	[get_ports "CH1_LPDDR4_1_0_ck_c_b[0]"] ;
set_property	PACKAGE_PIN	AR36	[get_ports "CH1_LPDDR4_1_0_ck_t_a[0]"] ;
set_property	PACKAGE_PIN	BD37	[get_ports "CH1_LPDDR4_1_0_ck_t_b[0]"] ;
set_property	PACKAGE_PIN	AP38	[get_ports "CH1_LPDDR4_1_0_cke_a[0]"] ;
set_property	PACKAGE_PIN	BD38	[get_ports "CH1_LPDDR4_1_0_cke_b[0]"] ;
set_property	PACKAGE_PIN	AT38	[get_ports "CH1_LPDDR4_1_0_cs_a[0]"] ;
set_property	PACKAGE_PIN	AV37	[get_ports "CH1_LPDDR4_1_0_cs_b[0]"] ;
set_property	PACKAGE_PIN	AW36	[get_ports "CH1_LPDDR4_1_0_dmi_a[0]"] ;
set_property	PACKAGE_PIN	BD33	[get_ports "CH1_LPDDR4_1_0_dmi_a[1]"] ;
set_property	PACKAGE_PIN	BF39	[get_ports "CH1_LPDDR4_1_0_dmi_b[0]"] ;
set_property	PACKAGE_PIN	AR33	[get_ports "CH1_LPDDR4_1_0_dmi_b[1]"] ;
set_property	PACKAGE_PIN	AW39	[get_ports "CH1_LPDDR4_1_0_dq_a[0]"] ;
set_property	PACKAGE_PIN	AY39	[get_ports "CH1_LPDDR4_1_0_dq_a[1]"] ;
set_property	PACKAGE_PIN	BF34	[get_ports "CH1_LPDDR4_1_0_dq_a[10]"] ;
set_property	PACKAGE_PIN	BG34	[get_ports "CH1_LPDDR4_1_0_dq_a[11]"] ;
set_property	PACKAGE_PIN	BB34	[get_ports "CH1_LPDDR4_1_0_dq_a[12]"] ;
set_property	PACKAGE_PIN	BC33	[get_ports "CH1_LPDDR4_1_0_dq_a[13]"] ;
set_property	PACKAGE_PIN	BD34	[get_ports "CH1_LPDDR4_1_0_dq_a[14]"] ;
set_property	PACKAGE_PIN	BD35	[get_ports "CH1_LPDDR4_1_0_dq_a[15]"] ;
set_property	PACKAGE_PIN	AU37	[get_ports "CH1_LPDDR4_1_0_dq_a[2]"] ;
set_property	PACKAGE_PIN	AU36	[get_ports "CH1_LPDDR4_1_0_dq_a[3]"] ;
set_property	PACKAGE_PIN	AU35	[get_ports "CH1_LPDDR4_1_0_dq_a[4]"] ;
set_property	PACKAGE_PIN	AV35	[get_ports "CH1_LPDDR4_1_0_dq_a[5]"] ;
set_property	PACKAGE_PIN	AW35	[get_ports "CH1_LPDDR4_1_0_dq_a[6]"] ;
set_property	PACKAGE_PIN	AY35	[get_ports "CH1_LPDDR4_1_0_dq_a[7]"] ;
set_property	PACKAGE_PIN	BG35	[get_ports "CH1_LPDDR4_1_0_dq_a[8]"] ;
set_property	PACKAGE_PIN	BE35	[get_ports "CH1_LPDDR4_1_0_dq_a[9]"] ;
set_property	PACKAGE_PIN	BG40	[get_ports "CH1_LPDDR4_1_0_dq_b[0]"] ;
set_property	PACKAGE_PIN	BE40	[get_ports "CH1_LPDDR4_1_0_dq_b[1]"] ;
set_property	PACKAGE_PIN	AP33	[get_ports "CH1_LPDDR4_1_0_dq_b[10]"] ;
set_property	PACKAGE_PIN	AN34	[get_ports "CH1_LPDDR4_1_0_dq_b[11]"] ;
set_property	PACKAGE_PIN	AP34	[get_ports "CH1_LPDDR4_1_0_dq_b[12]"] ;
set_property	PACKAGE_PIN	AR32	[get_ports "CH1_LPDDR4_1_0_dq_b[13]"] ;
set_property	PACKAGE_PIN	AU33	[get_ports "CH1_LPDDR4_1_0_dq_b[14]"] ;
set_property	PACKAGE_PIN	AV34	[get_ports "CH1_LPDDR4_1_0_dq_b[15]"] ;
set_property	PACKAGE_PIN	BF37	[get_ports "CH1_LPDDR4_1_0_dq_b[2]"] ;
set_property	PACKAGE_PIN	BF36	[get_ports "CH1_LPDDR4_1_0_dq_b[3]"] ;
set_property	PACKAGE_PIN	BE36	[get_ports "CH1_LPDDR4_1_0_dq_b[4]"] ;
set_property	PACKAGE_PIN	BG36	[get_ports "CH1_LPDDR4_1_0_dq_b[5]"] ;
set_property	PACKAGE_PIN	BE39	[get_ports "CH1_LPDDR4_1_0_dq_b[6]"] ;
set_property	PACKAGE_PIN	BG39	[get_ports "CH1_LPDDR4_1_0_dq_b[7]"] ;
set_property	PACKAGE_PIN	AW33	[get_ports "CH1_LPDDR4_1_0_dq_b[8]"] ;
set_property	PACKAGE_PIN	AV33	[get_ports "CH1_LPDDR4_1_0_dq_b[9]"] ;

set_property	PACKAGE_PIN	AW37	[get_ports "CH1_LPDDR4_1_0_dqs_c_a[0]"] ;
set_property	PACKAGE_PIN	AV38	[get_ports "CH1_LPDDR4_1_0_dqs_t_a[0]"] ;

set_property	PACKAGE_PIN	BG33	[get_ports "CH1_LPDDR4_1_0_dqs_c_a[1]"] ;
set_property	PACKAGE_PIN	BF33	[get_ports "CH1_LPDDR4_1_0_dqs_t_a[1]"] ;

set_property	PACKAGE_PIN	BF38	[get_ports "CH1_LPDDR4_1_0_dqs_c_b[0]"] ;
set_property	PACKAGE_PIN	BE37	[get_ports "CH1_LPDDR4_1_0_dqs_t_b[0]"] ;

set_property	PACKAGE_PIN	AT32	[get_ports "CH1_LPDDR4_1_0_dqs_c_b[1]"] ;
set_property	PACKAGE_PIN	AU32	[get_ports "CH1_LPDDR4_1_0_dqs_t_b[1]"] ;

set_property	PACKAGE_PIN	AW28	[get_ports "CH1_LPDDR4_1_0_reset_n[0]"] ;
