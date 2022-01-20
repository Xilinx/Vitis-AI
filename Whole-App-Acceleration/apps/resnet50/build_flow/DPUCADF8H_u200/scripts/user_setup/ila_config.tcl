
#set ILA_INSTNAME ila_img_ctl
#set PROB_NUM 5
#set PROB_WIDTH {4 4 4 4 4}
#gen_ila $ILA_INSTNAME $PROB_NUM $PROB_WIDTH

#set ILA_INSTNAME ila_org_dma
#set PROB_NUM 21
#set PROB_WIDTH {9 9 9 9 1 1 1 1 9 9 9 9 1 1 1 1 1 1 1 1 6}
#gen_ila $ILA_INSTNAME $PROB_NUM $PROB_WIDTH

#set ILA_INSTNAME ila_org_dma_top
#set PROB_NUM 22
#set PROB_WIDTH {32 32 32 32 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1}
#gen_ila $ILA_INSTNAME $PROB_NUM $PROB_WIDTH

#set ILA_INSTNAME ila_req_ctl
#set PROB_NUM 4
#set PROB_WIDTH {32 1 1 1}
#gen_ila $ILA_INSTNAME $PROB_NUM $PROB_WIDTH

#set ILA_INSTNAME ila_wreq_ctl
#set PROB_NUM 1
#set PROB_WIDTH {6}
#gen_ila $ILA_INSTNAME $PROB_NUM $PROB_WIDTH



