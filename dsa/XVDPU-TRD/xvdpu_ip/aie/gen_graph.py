#! /usr/bin/env python3
import sys
import os
import json

#enable option
PL_FREQ=125
VVER=1922
#constraint core placement 
BATCH0_ROW_OFFSET=0;
BATCH0_COL_OFFSET=6;
COL_INTERVAL=0;
BAT_SHAREWGT=1;
MAX_BAT_PER_SHAREWGT=3

#file name
PARAM_FILE = "./src/param.h"
GRAPH_FILE = "./src/graph_conv.h"
PORTCONSTRAINT_FILE = "./src/port_constraint.json"
PLATFORM_FILE = "./src/graph_conv.cc"

#port prefix used in graph_conv.h
PORTA_PREFIX = "datain_a"
PORTB_PREFIX = "datain_b"
PORTC_PREFIX = "dataout"

#PLIO prefix used in graph_conv.cc
PLIOA_PREFIX = "aPr_"
PLIOB_PREFIX = "bPr_"
PLIOC_PREFIX = "cPr_"
GSIZE = 4

#parameters will be set according to core num
#NUM_CORE = NUM_BATCH * NUM_CHAIN * LEN_CHAIN
NUM_CORE = 64
#num of batch
NUM_BATCH = 1
#num of chains to expand pixel & outputchannel parallel
NUM_CHAIN = 32
# num of ME in one chain, expand input channel parallel
LEN_CHAIN = 2
#port num for each batch
NUM_PORT_A = 8
NUM_PORT_B = 16
NUM_PORT_C = 32
PORT_B_BAT = NUM_BATCH
#port width for each batch
WIDTH_PORT_A = 128
WIDTH_PORT_B = 128
WIDTH_PORT_C = 64

# lookup table for generate port_constraint: 
# PLIOA_IDX2COL[0] is 0, PLIOA[0] will be set to col (0 + BATCH0_COL_OFFSET)
PLIOA_IDX2COL = [ 0,  1, 2, 3, 4, 5, 6, 7]
PLIOB_IDX2COL = [ 0,  1,  0,  1, 
                  2,  3,  2,  3,
                  4,  5,  4,  5,
                  6,  7,  6,  7]
PLIOC_IDX2COL = [1, 0, 1, 0, 1, 0, 1, 0,
                 3, 2, 3, 2, 3, 2, 3, 2,
                 5, 4, 5, 4, 5, 4, 5, 4,
                 7, 6, 7, 6, 7, 6, 7, 6]

# lookup table for generate: kernel idx is the key
# kernel 0, idx is 0, CORE2PORTA[0] is 0, so aPr[0] will connect to kernel[0]
CORE2PORTA = [ 0, 1, 0, 1, 2, 3, 2, 3, 4, 5, 4, 5, 6, 7, 6, 7,
               0, 1, 0, 1, 2, 3, 2, 3, 4, 5, 4, 5, 6, 7, 6, 7,
               0, 1, 0, 1, 2, 3, 2, 3, 4, 5, 4, 5, 6, 7, 6, 7,
               0, 1, 0, 1, 2, 3, 2, 3, 4, 5, 4, 5, 6, 7, 6, 7]

CORE2PORTB = [  0,  1,  2,  3,  0,  1,  2,  3,  0,  1,  2,  3,  0,  1,  2,  3,
                4,  5,  6,  7,  4,  5,  6,  7,  4,  5,  6,  7,  4,  5,  6,  7,
                8,  9, 10, 11,  8,  9, 10, 11,  8,  9, 10, 11,  8,  9, 10, 11,
               12, 13, 14, 15, 12, 13, 14, 15, 12, 13, 14, 15, 12, 13, 14, 15]

#for PORTC, if the port is cascade stream, it diect to destination core
CORE2PORTC = [  1,   0, 1,   2,  5,   2, 3,   6,  9,   4, 5,  10, 13,   6, 7,  14,
               17,   8, 9,  18, 21,  10,11,  22, 25,  12,13,  26, 29,  14,15,  30,
               33,  16,17,  34, 37,  18,19,  38, 41,  20,21,  42, 45,  22,23,  46,
               49,  24,25,  50, 53,  26,27,  54, 57,  28,29,  58, 61,  30,31,  62]

#constraint buffer size in Bytes
MAX_BYTES_BUFA = 1024
MAX_BYTES_BUFB = 2048
MAX_BYTES_BUFC = 1024
INTER_OFFSET = 1024
LAYER_PARAM_BUF_SIZE = 16
WLUT_BUF_SIZE = 64


#constraint buffer address
STACK_OFFSET = "0x3800"
BUFA_PING_OFFSET = "0x0000"
BUFA_PONG_OFFSET = "0x4000"
BUFB_PING_OFFSET = "0x1000"
BUFB_PONG_OFFSET = "0x5000"
BUFC_PING_OFFSET = "0x2000"
BUFC_PONG_OFFSET = "0x6000"

def getCommitId():
	commit_id = "0d9ca43e"
	commit_id = commit_id.strip()
	assert(commit_id != ''), "getCommitId error"
	return commit_id
def splitCMI(commit_id):
	cmi_list = []
	cmi_list.append(commit_id[4: 8])
	cmi_list.append(commit_id[0: 4])
	return cmi_list

def enableConfig():
	retStr = '\n'
	retStr += "#define AIE_ACTIVATIONS_EN 0\n"
	retStr += "//[deprecated]\n#define CONV_LEAKYRELU_EN 0\n"
	retStr += "\n#define PL_FREQ "+str(PL_FREQ)+"\n"
	retStr += "#define VVER "+str(VVER)+"\n"
	return retStr

def genParamHPP():
    fo = open(PARAM_FILE, "w")
    word = ""		
    word += "#ifndef __PARAM_H__\n"
    word += "#define __PARAM_H__\n\n"
    cmi = getCommitId()
    cmi_list = splitCMI(cmi)
    for i in range(0, len(cmi_list)):
      word += "#define COMMIT_ID_"+str(i)+" 0x"+ cmi_list[i] + "\n"
    word += "#define CORE_PER_BATCH "+str(NUM_CORE)+"\n"
    word += "#define NUM_BATCH "+str(NUM_BATCH)+"\n"
    word += "\n#define MAX_BYTES_BUFA "+str(MAX_BYTES_BUFA)+"\n"
    word += "#define MAX_BYTES_BUFB "+str(MAX_BYTES_BUFB)+"\n"
    word += "#define MAX_BYTES_BUFC "+str(MAX_BYTES_BUFC)+"\n"
    word += "#define INTER_OFFSET "+str(INTER_OFFSET)+"\n\n"
    word += "#define LAYER_PARAM_BUF_SIZE "+str(LAYER_PARAM_BUF_SIZE)+"\n"
    word += "#define WLUT_BUF_SIZE "+str(WLUT_BUF_SIZE)+"\n\n"
    word += "enum CHIPTYPE{\n"
    word += "ES1=0,\n"
    word += "ES2=1,\n"
    word += "PROD=2};\n"
    word += "#define CHIP PROD\n\n"
    word += enableConfig()
    word += "\n"
    word += "#endif /*__PARAM_H__*/\n"
    fo.write(word)
    fo.close()


def genKernelDef():
  batch_cores = NUM_CHAIN*LEN_CHAIN
  ret = "\t//kernel declare\n "
  ret += "\t//split kernel declare\n"
  ret += (
            u"\tfor(int idx_sp=0; idx_sp<"+ str(NUM_BATCH*batch_cores)+"; idx_sp++){\n"
            u"\t\tsplit[idx_sp] = pktsplit<2>::create();\n\t}\n"
         )
  ret += "\t//ctrl kernel declare\n"
  ret += (
            u"\tfor(int idxctl=0; idxctl<"+ str(NUM_BATCH*batch_cores)+"; idxctl++){\n"
            u"\t\tctrl[idxctl] = pktcontrol::create();\n\t}\n"
         )
  ret += "\n\t//conv kernel declare\n "
  for idx_bat in range(0, int(NUM_BATCH)):
    ret += "\t//batch "+str(idx_bat) + "\n"
    for idx_ker in range(0, batch_cores):
      idxStr = "[" + str(idx_bat*batch_cores+idx_ker) + "]"
      ret += "\t//kernel declare core" + idxStr+"\n"
      if (idx_ker%GSIZE == 0 or idx_ker%GSIZE == 3):
          ret += "\tsuperkernel" + idxStr + " = kernel::create(super_kernel_input_casc);\n"
          ret += "\tsource(superkernel" + idxStr +")  = \"kernels/super_kernel_input_casc.cc\";\n"
      elif (idx_ker%GSIZE == 1 or idx_ker%GSIZE ==2):
          ret += "\tsuperkernel" + idxStr + " = kernel::create(super_kernel_casc_output);\n"
          ret += "\tsource(superkernel" + idxStr +")  = \"kernels/super_kernel_casc_output.cc\";\n"
      ret += "\truntime<ratio>(superkernel" + idxStr  + ") = 0.9;\n\n"
      idxParamStrPing =  "[" +str(6*(idx_bat*batch_cores+idx_ker)+0) + "]"
      idxParamStrPong =  "[" +str(6*(idx_bat*batch_cores+idx_ker)+1) + "]"
      idxParamStrCachepi =  "[" +str(6*(idx_bat*batch_cores+idx_ker)+2) + "]"
      idxParamStrCachepo =  "[" +str(6*(idx_bat*batch_cores+idx_ker)+3) + "]"
      idxParamStrPping =  "[" +str(6*(idx_bat*batch_cores+idx_ker)+4) + "]"
      idxParamStrPpong =  "[" +str(6*(idx_bat*batch_cores+idx_ker)+5) + "]"
      ret += "\tpmtr" + idxParamStrPing + " = parameter::array(wlut_ping);\n"
      ret += "\tpmtr" + idxParamStrPong + " = parameter::array(wlut_pong);\n"
      ret += "\tpmtr" + idxParamStrCachepi + " = parameter::array(wlut_cache_ping);\n"
      ret += "\tpmtr" + idxParamStrCachepo + " = parameter::array(wlut_cache_pong);\n"
      ret += "\tpmtr" + idxParamStrPping + " = parameter::array(layer_param_buf_ping);\n"
      ret += "\tpmtr" + idxParamStrPpong + " = parameter::array(layer_param_buf_pong);\n"
  return ret


def getCoreRowCol(idx_bat, idx_ker):
    col_per_bat = (NUM_PORT_B//2 + COL_INTERVAL)
    strCol = str(int(BATCH0_COL_OFFSET +idx_bat*col_per_bat + (idx_ker//16)*2 + idx_ker%2));
    strRow = str(int(BATCH0_ROW_OFFSET +(idx_ker%16)//2));
    return strRow, strCol
    
def genConnect():
  batch_cores = NUM_CHAIN*LEN_CHAIN
  ret = "\n\t//declare connection\n"
  col_per_batch = NUM_PORT_B//2
  for idx_bat in range(0, int(NUM_BATCH)):
    ret += "\t//batch "+str(idx_bat) + "\n"
    base_a = idx_bat * NUM_PORT_A 
    base_b = (idx_bat//MAX_BAT_PER_SHAREWGT) * NUM_PORT_B
    base_c = idx_bat * NUM_PORT_C
    base_k = idx_bat * batch_cores
    for idx_ker in range(0, batch_cores):
      idxStr = "[" + str(base_k + idx_ker) + "]"
      idxParamStrPing = "[" + str(6*(base_k + idx_ker)+0) + "]"
      idxParamStrPong = "[" + str(6*(base_k + idx_ker)+1) + "]"
      idxParamStrCachepi = "[" + str(6*(base_k + idx_ker)+2) + "]"
      idxParamStrCachepo = "[" + str(6*(base_k + idx_ker)+3) + "]"
      idxParamStrPping = "[" + str(6*(base_k + idx_ker)+4) + "]"
      idxParamStrPpong = "[" + str(6*(base_k + idx_ker)+5) + "]"
      ret += "\t//core"+idxStr+" connection and buf constraint\n"
      #PORTA -> conv.in[0]
      aidx = base_a + CORE2PORTA[idx_ker]
      ret += "\t//" + PORTA_PREFIX + "["+str(aidx) + "] --> core" +idxStr + "\n"
      ret += "\tconnect<stream, window<MAX_BYTES_BUFA>>("+PORTA_PREFIX+"[" + str(aidx) + "], async(superkernel" + idxStr + ".in[0]));\n"
      #PORTB -> split
      bidx = base_b + CORE2PORTB[idx_ker]
      ret += "\t//" + PORTB_PREFIX + "["+str(bidx) + "] --> core" +idxStr + "\n"
      ret += "\tconnect<pktstream >("+PORTB_PREFIX+"[" + str(bidx) + "], split" + idxStr + ".in[0]);\n"
      #split.out[0] -> conv.in[1]
      ret += "\tconnect<pktstream, window<MAX_BYTES_BUFB> >(split" + idxStr + ".out[0], async(superkernel" + idxStr + ".in[1]));\n"
      #split.out[1] -> ctrl.in[0]
      ret += "\tconnect<pktstream, pktstream>(split" + idxStr + ".out[1], ctrl" + idxStr + ".in[0]);\n"
      ret += "\tutilization(split"+idxStr+".out[0]) = 0.9;\n"
      ret += "\tutilization(split"+idxStr+".out[1]) = 0.1;\n"
      strSuperKer = "superkernel" + idxStr
      if (idx_ker%4 == 0 or idx_ker%4==3):
        #conv.out[0] -> cascade-> nextconv.in[2]
        ret += "\t//core" + idxStr +" --> [cascade]\n"
        ret += "\tconnect<cascade>(superkernel" + idxStr + ".out[0], superkernel[" + str(base_k + CORE2PORTC[idx_ker]) +"].in[2]);\n"

        ret +="#if(VVER==212)\n"
        ret += "\tdisable_dma_autostart(superkernel" + idxStr + ".in[0]);\n"
        ret += "\tdisable_dma_autostart(superkernel" + idxStr + ".in[1]);\n"
        ret +="#endif\n"
        #cascTo = 1 if (idx_ker%4==0) else -1;
        #ret += "\tconnect<cascade>(superkernel" + idxStr + ".out[0], superkernel[" + str(idx_ker+cascTo) +"].in[2]);\n"
      elif (idx_ker%4== 1 or idx_ker%4==2):
        #cascadeconv.out[0] -> PORTC
        #cidx = base_c + (idx_ker/16)*8 + (idx_ker%16)/2
        cidx = base_c + CORE2PORTC[idx_ker]
        ret += "\t//core" + idxStr +" --> " + PORTC_PREFIX + "["+str(cidx) + "]\n"
        ret += "\tconnect<window<MAX_BYTES_BUFC>, pktstream>(async(superkernel" + idxStr + ".out[0]), "+PORTC_PREFIX+"[" + str(cidx) +"]);\n"
        ret += "\tlocation<kernel>("+strSuperKer+") = location<buffer>("+strSuperKer+".out[0]);\n"        
        ret += "\tlocation<buffer>("+strSuperKer+".out[0]) = {offset("+BUFC_PING_OFFSET+"), offset("+BUFC_PONG_OFFSET+")};\n"        
        ret +="#if(VVER==212)\n"
        ret += "\tdisable_dma_autostart(superkernel" + idxStr + ".in[0]);\n"
        ret += "\tdisable_dma_autostart(superkernel" + idxStr + ".in[1]);\n"
        ret += "\tdisable_dma_autostart(superkernel" + idxStr + ".out[0]);\n"
        ret +="#endif\n"
      ret += "\tconnect<>(pmtr"+idxParamStrPing+","+strSuperKer+");\n"
      ret += "\tconnect<>(pmtr"+idxParamStrPong+","+strSuperKer+");\n"
      ret += "\tconnect<>(pmtr"+idxParamStrCachepi+","+strSuperKer+");\n"
      ret += "\tconnect<>(pmtr"+idxParamStrCachepo+","+strSuperKer+");\n"
      ret += "\tconnect<>(pmtr"+idxParamStrPping+","+strSuperKer+");\n"
      ret += "\tconnect<>(pmtr"+idxParamStrPpong+","+strSuperKer+");\n"
      strRow, strCol = getCoreRowCol(idx_bat, idx_ker)
      ret += "\tlocation<kernel>("+strSuperKer+") = tile("+ strCol + ", " + strRow + ");\n"

      ret += "\tlocation<parameter>(pmtr"+idxParamStrPing+") = address("+ strCol + ", " + strRow + ",16384-WLUT_BUF_SIZE*4);\n"
      ret += "\tlocation<parameter>(pmtr"+idxParamStrPong+") = address("+ strCol + ", " + strRow + ",32768-WLUT_BUF_SIZE*4);\n"
      ret += "\tlocation<parameter>(pmtr"+idxParamStrCachepi+") = address("+ strCol + ", " + strRow + ",32768-WLUT_BUF_SIZE*4-WLUT_BUF_SIZE*4);\n"
      ret += "\tlocation<parameter>(pmtr"+idxParamStrCachepo+") = address("+ strCol + ", " + strRow + ",32768-WLUT_BUF_SIZE*4-WLUT_BUF_SIZE*8);\n"
      ret += "\tlocation<parameter>(pmtr"+idxParamStrPping+") = address("+ strCol + ", " + strRow + ",32768-WLUT_BUF_SIZE*4-WLUT_BUF_SIZE*8-LAYER_PARAM_BUF_SIZE*4);\n"
      ret += "\tlocation<parameter>(pmtr"+idxParamStrPpong+") = address("+ strCol + ", " + strRow + ",32768-WLUT_BUF_SIZE*4-WLUT_BUF_SIZE*8-LAYER_PARAM_BUF_SIZE*8);\n"
      ret += "\tlocation<kernel>("+strSuperKer+") = location<interconnect>(ctrl"+idxStr+");\n"
      ret += "\tlocation<kernel>("+strSuperKer+") = location<stack>(" + strSuperKer+");\n"        
      ret += "\tlocation<stack>("+strSuperKer+")= offset(" + STACK_OFFSET + ");\n"        
      ret += "\tlocation<kernel>("+strSuperKer+") = location<buffer>("+strSuperKer+".in[0]);\n"        
      ret += "\tlocation<buffer>("+strSuperKer+".in[0]) = {offset("+BUFA_PING_OFFSET+"), offset("+BUFA_PONG_OFFSET+")};\n"        
      ret += "\tlocation<kernel>("+strSuperKer+") = location<buffer>("+strSuperKer+".in[1]);\n"        
      ret += "\tlocation<buffer>("+strSuperKer+".in[1]) = {offset("+BUFB_PING_OFFSET+"), offset("+BUFB_PONG_OFFSET+")};\n\n"        
  return ret

def genGraph():
	fo = open(GRAPH_FILE, "w")
	word = ""		
	word += "#ifndef __CARDANO_GRAPH_H__\n"
	word += "#define __CARDANO_GRAPH_H__\n"
	word += "#include \"kernels_conv.h\"\n"
	word += "#include \"param.h\"\n"
	word += "\n#if(VVER!=1922 && VVER != 201)\n"
	word += "#include <adf.h>\n\n"
	word += "using namespace adf;\n\n"
	word += "#else\n\n"
	word += "#include <cardano.h>\n\n"
	word += "using namespace cardano;\n\n"
	word += "#endif\n\n"
	word += "extern int wlut_pong[WLUT_BUF_SIZE];\n"
	word += "extern int wlut_ping[WLUT_BUF_SIZE];\n"
	word += "extern int wlut_cache_ping[WLUT_BUF_SIZE];\n"
	word += "extern int wlut_cache_pong[WLUT_BUF_SIZE];\n"
	word += "extern int layer_param_buf_ping[LAYER_PARAM_BUF_SIZE];\n"
	word += "extern int layer_param_buf_pong[LAYER_PARAM_BUF_SIZE];\n"
	word += "class CONV_CHAINS : public graph {\n"
	word += "private:\n"
	word += "\tkernel superkernel["+str(NUM_CORE*NUM_BATCH)+"];\n"
	word += "\tpktsplit<2> split["+str(NUM_CORE*NUM_BATCH)+"];\n"
	word += "\tpktcontrol ctrl["+str(NUM_CORE*NUM_BATCH)+"];\n"
	word += "\tparameter pmtr["+str(NUM_CORE*NUM_BATCH*6)+"];\n"
	word += "public:\n"
	word += "\tport<input> "+PORTA_PREFIX+"["+str(NUM_BATCH*NUM_PORT_A)+"];\n"
	word += "\tport<input> "+PORTB_PREFIX+"["+str(PORT_B_BAT*NUM_PORT_B)+"];\n"
	word += "\tport<output> "+PORTC_PREFIX+"["+str(NUM_BATCH*NUM_PORT_C)+"];\n\n"
	word += "CONV_CHAINS() {\n"
	fo.write(word)

	fo.write(genKernelDef())
	fo.write(genConnect())
	word = ""
	word +="  }\n"
	word +="};\n"
	word +="#endif\n"
	fo.write(word)
	fo.close()

def getNameA(idx_bat, idx_row, idx_col):
    return "b"+str(idx_bat) + "_r" + str(idx_row)+"_c"+str(idx_col)

def getNameB(idx_bat, idx_group, idx_len):
    return "b" + str(idx_bat) + "_g"+str(idx_group) + "_l" + str(idx_len)

def getNameC(idx_bat, idx_num, idx_row, idx_col):
    return "b"+str(idx_bat) + "_n"+str(idx_num)+"_r"+str(idx_row) +"_c"+ str(idx_col)
    
def genIOdefine():
  ret = "\n"
  #port_num = #PORTA + #PORTB + #PORTC
  plat = "\nsimulation::platform<" + str(NUM_BATCH*NUM_PORT_A\
  + PORT_B_BAT*NUM_PORT_B) \
  + "," + str(NUM_BATCH*NUM_PORT_C) + "> platform("
  #input A
  ret += "//PORTA\n"
  for idx_bat in range(0, int(NUM_BATCH)):
    for idx_aPr in range(0, NUM_PORT_A):
      #idxStr = "b"+str(idx_bat) + "_r" + str(idx_aPr/2*2)+"_c"+str(idx_aPr%2)
      idxStr = getNameA(idx_bat, int(idx_aPr//2*2), idx_aPr%2)
      ret += "PLIO *" + PLIOA_PREFIX + idxStr + "  = new PLIO(\"XvDPU_v1/M_AXIS_PL2ME"+ idxStr +"\", plio_"+str(WIDTH_PORT_A)+"_bits, \"./data/" + PLIOA_PREFIX + idxStr + ".txt\");\n";
      plat += PLIOA_PREFIX + str(idxStr) + ",\n\t"
  #input B
  ret += "\n//PORTB\n"
  for idx_bat in range(0, int(PORT_B_BAT)):
    for idx_bPr in range(0, NUM_PORT_B):
      #idxStr = "g"+str(idx_bPr/2) + "_l" + str(idx_bPr%2)
      idxStr = getNameB(idx_bat, int(idx_bPr//2), idx_bPr%2)
      ret += "PLIO *" + PLIOB_PREFIX + idxStr +"  = new PLIO(\"XvDPU_v1/M_AXIS_WM2ME" + idxStr +"\", plio_"+str(WIDTH_PORT_B)+"_bits, \"./data/" +PLIOB_PREFIX + idxStr + ".txt\");\n";
      plat += PLIOB_PREFIX + idxStr + ", \n\t"
  #output C
  ret += "\n//PORTC\n"
  for idx_bat in range(0, int(NUM_BATCH)):
    for idx_cPr in range(0, NUM_PORT_C):
      #idxStr = "b"+str(idx_bat) + "_n"+str(idx_cPr/8)+"_r"+str(idx_cPr%8) +"_c"+ str(1 if idx_cPr%2==0 else 0)
      idx_col = 1 if (idx_cPr%2 == 0) else 0
      idxStr = getNameC(idx_bat, int(idx_cPr//8), idx_cPr%8, idx_col)
      #ret += "PLIO *" + PLIOC_PREFIX + idxStr +"  = new PLIO(\"XvDPU_v1/M_AXIS_ME2PL"+idxStr + "\", plio_"+str(WIDTH_PORT_C)+"_bits, \"./output/" + PLIOC_PREFIX + idxStr + ".txt\");\n";
      OCPG=1 if NUM_PORT_C==1 else int((NUM_PORT_C*2)//8) 
      OCPR=1 if NUM_PORT_C==1 else int((NUM_PORT_C*2)//16)
      idxStrInPlat = getNameC(idx_bat, int((idx_cPr%OCPG)//2%OCPR), int((idx_cPr//OCPG)*2) + (idx_cPr%OCPG)%2, 1-idx_cPr%2)
      ret += "PLIO *" + PLIOC_PREFIX + idxStrInPlat +"  = new PLIO(\"XvDPU_v1/M_AXIS_ME2PL"+idxStrInPlat + "\", plio_"+str(WIDTH_PORT_C)+"_bits, \"./output/" + PLIOC_PREFIX + idxStrInPlat + ".txt\");\n";
      plat += PLIOC_PREFIX + idxStrInPlat + (", \n\t" if ((idx_bat*NUM_PORT_C+ idx_cPr) != (int(NUM_BATCH*NUM_PORT_C)-1)) else "")
  plat += ");"
  ret += plat
  return ret

def genIOConnect():
  ret = ""
  ret += "//A\n"
  for idx_bat in range(0, int(NUM_BATCH)):
    for idx_aPr in range(0, NUM_PORT_A):
      idx_A = idx_bat*NUM_PORT_A+ idx_aPr
      ret += "connect<> net_A"+ str(idx_A) + "(platform.src["+ str(idx_A) + "], convgraph." + PORTA_PREFIX+"[" + str(idx_A) +"]);\n"
  
  ret += "//B\n"
  for idx_bat in range(0, int(PORT_B_BAT)):
    base_b = NUM_BATCH*NUM_PORT_A + idx_bat*NUM_PORT_B
    for idx_bPr in range(0, NUM_PORT_B):
      platIdxStr = str(base_b + idx_bPr)
      idxStr = str(idx_bat*NUM_PORT_B + idx_bPr)
      ret += "connect<pktstream> net_B"+ idxStr + "(platform.src["+ platIdxStr + "], convgraph." + PORTB_PREFIX+"[" + idxStr +"]);\n"
  
  ret += "//C\n"
  OCPG=1 if NUM_PORT_C==1 else int((NUM_PORT_C*2)//8)
  OCPR=1 if NUM_PORT_C==1 else int((NUM_PORT_C*2)//16)
  for idx_bat in range(0, int(NUM_BATCH)):
    for idx_cPr in range(0, int(NUM_PORT_C)):
      idxStr = str(idx_bat*NUM_PORT_C + idx_cPr)
      #n*OCPR + r;
      idxPlatStr = str(int(idx_bat*NUM_PORT_C + ((idx_cPr%OCPG))//2%OCPR*8+  ((idx_cPr//OCPG)*2 + (idx_cPr%OCPG)%2)));
      ret += "connect<> net_C" + idxStr + "(convgraph."+PORTC_PREFIX+"["+idxPlatStr+"]" + ", platform.sink[" + idxStr+ "]);\n"
  return ret

def genPlatform():
	fo = open(PLATFORM_FILE, "w")
	word = ""		
	word += "#include \"graph_conv.h\"\n"
	fo.write(word)

	fo.write(genIOdefine())
	
	fo.write("\n\nCONV_CHAINS convgraph;\n\n")

	fo.write(genIOConnect())

	word ="\n"
	word+="int main(int argc, char ** argv) {\n"
	word+="  convgraph.init();\n"
	word+="  convgraph.run();\n"
	word+="  convgraph.end();\n"
	word+="  return 0;\n"
	word+="}\n"
	fo.write(word)
	fo.close()

def getConstEntry(name, col):
  word = "\t\t\""+name+"\": {\n"
  word += "\t\t\t\"shim\": {\n"
  word += "\t\t\t\t\"column\": " + str(col)+"\n"
  word += "\t\t\t}\n"
  word += "\t\t},\n"
  return word

def genPortConstraint():
  word = ""	
  word += "{\n"
  word += "\t\"NodeConstraints\": {\n"
  #input A
  COL_MAX=44
  col_per_bat = (NUM_PORT_B//2+ COL_INTERVAL)
  if PL_FREQ==125:
    for idx_bat in range(0, int(NUM_BATCH)):
      for idx_aPr in range(0, NUM_PORT_A): 
        col = BATCH0_COL_OFFSET + idx_bat*col_per_bat+ PLIOA_IDX2COL[idx_aPr]
        if col > COL_MAX : continue
        idxStr = "b"+str(idx_bat) + "_r" + str(int(idx_aPr//2*2))+"_c"+str(idx_aPr%2)
        word += getConstEntry(PLIOA_PREFIX + idxStr, col)
    #input B
    if MAX_BAT_PER_SHAREWGT == 1:
     for idx_bat in range(0, int(NUM_BATCH)):
      for idx_bPr in range(0, NUM_PORT_B):
        col = BATCH0_COL_OFFSET + idx_bat*col_per_bat + PLIOB_IDX2COL[idx_bPr]
        if col > COL_MAX : continue
        idxStr = getNameB(idx_bat, int(idx_bPr//2), idx_bPr%2)
        word += getConstEntry(PLIOB_PREFIX + idxStr, col)
    elif NUM_BATCH == 8 and BAT_SHAREWGT==4:
     for idx_bat in range(0, 3):
      for idx_bPr in range(0, NUM_PORT_B):
        col = 6 + idx_bat*12+ idx_bPr
        if col > COL_MAX : continue
        idxStr = getNameB(idx_bat, int(idx_bPr//2), idx_bPr%2)
        word += getConstEntry(PLIOB_PREFIX + idxStr, col)
    #output C
    for idx_bat in range(0, int(NUM_BATCH)):
      for idx_cPr in range(0, NUM_PORT_C):
        col = BATCH0_COL_OFFSET + idx_bat*col_per_bat+ PLIOC_IDX2COL[idx_cPr]
        if col > COL_MAX : continue
        idxStr = "b"+str(idx_bat) + "_n"+str(int(idx_cPr//8))+"_r"+str(idx_cPr%8) +"_c"+ str(1 if idx_cPr%2==0 else 0)
        word += getConstEntry(PLIOC_PREFIX + idxStr, col)
    word = word[:-2]
    word += "\n"
  elif PL_FREQ == 250:
    for idx_bat in range(0, int(NUM_BATCH)):
      for idx_aPr in range(0, NUM_PORT_A): 
        col = BATCH0_COL_OFFSET + idx_bat*col_per_bat+ PLIOA_IDX2COL[idx_aPr]
        if col > COL_MAX : continue
        idxStr = "b"+str(idx_bat) + "_r" + str(int(idx_aPr//2*2))+"_c"+str(idx_aPr%2)
        word += getConstEntry(PLIOA_PREFIX + idxStr, col)
    #output C
    for idx_bat in range(0, int(NUM_BATCH)):
      for idx_cPr in range(0, NUM_PORT_C):
        col = BATCH0_COL_OFFSET + idx_bat*col_per_bat+ PLIOC_IDX2COL[idx_cPr]
        if col > COL_MAX : continue
        idxStr = "b"+str(idx_bat) + "_n"+str(int(idx_cPr//8))+"_r"+str(idx_cPr%8) +"_c"+ str(1 if idx_cPr%2==0 else 0)
        word += getConstEntry(PLIOC_PREFIX + idxStr, col)
    word = word[:-2]
    word += "\n"
    '''
  elif PL_FREQ == 250:
    if NUM_CORE == 32 and (NUM_BATCH == 1 or NUM_BATCH == 3) and (COL_INTERVAL != 0):
      coreColStart = 18 if NUM_BATCH == 1 else 10
      interVal = 4
      for idx_bat in range(0, int(NUM_BATCH)):
        PLIOA_IDX2COL_NEW = [4, 5, 4, 5, 0, 1, 0, 1]
        for idx_aPr in range(0, NUM_PORT_A): 
          col = coreColStart + idx_bat*(2*interVal) + PLIOA_IDX2COL_NEW[idx_aPr] 
          idxStr = "b"+str(idx_bat) + "_r" + str(int(idx_aPr//2*2))+"_c"+str(idx_aPr%2)
          word += getConstEntry(PLIOA_PREFIX + idxStr, col)
      for idx_bat in range(0, int(NUM_BATCH)):
        PLIOB_IDX2COL_NEW = [3, 6, 3, 6, -1, 2, -1, 2]
        for idx_bPr in range(0, NUM_PORT_B):
          col = coreColStart + idx_bat*(2*interVal)+ PLIOB_IDX2COL_NEW[idx_bPr]
          idxStr = getNameB(idx_bat, int(idx_bPr//2), idx_bPr%2)
          word += getConstEntry(PLIOB_PREFIX + idxStr, col)
      for idx_bat in range(0, int(NUM_BATCH)):
        PLIOC_IDX2COL_NEW = [1, 0, 1, 0, 2, -1, 2, -1, 5, 4, 5, 4, 6, 3, 6, 3]
        for idx_cPr in range(0, NUM_PORT_C):
          col = coreColStart + idx_bat*(2*interVal) + PLIOC_IDX2COL_NEW[idx_cPr]
          idxStr = "b"+str(idx_bat) + "_n"+str(int(idx_cPr//8))+"_r"+str(idx_cPr%8) +"_c"+ str(1 if idx_cPr%2==0 else 0)
          word += getConstEntry(PLIOC_PREFIX + idxStr, col)
      word = word[:-2]
      word += "\n"
      '''
  word+="\t}\n"
  word+="}\n"
  fo = open(PORTCONSTRAINT_FILE, "w")
  fo.write(word)
  fo.close()

def genConfig(corenum, batch_num):
    print("[INFO] generating graph: [" + str(batch_num)+" batch(s), " + str(corenum) + " cores/batch]")
    global NUM_CORE
    global NUM_BATCH
    global NUM_CHAIN
    global NUM_PORT_A
    global NUM_PORT_B
    global NUM_PORT_C
    global WIDTH_PORT_A
    global WIDTH_PORT_B
    global WIDTH_PORT_C
    global PLIOA_IDX2COL
    global PLIOB_IDX2COL
    global PLIOC_IDX2COL
    global CORE2PORTA
    global CORE2PORTB
    global CORE2PORTC
    global BATCH0_COL_OFFSET
    global COL_INTERVAL
    global PORT_B_BAT 
    global MAX_BAT_PER_SHAREWGT
    NUM_BATCH = batch_num
    PORT_B_BAT = NUM_BATCH//BAT_SHAREWGT
    PORT_B_BAT += 0 if NUM_BATCH%BAT_SHAREWGT==0 else 1
    MAX_BAT_PER_SHAREWGT = NUM_BATCH//PORT_B_BAT
    MAX_BAT_PER_SHAREWGT += 0 if NUM_BATCH%PORT_B_BAT==0 else 1
    if corenum == 2:
       NUM_CORE = 2
       NUM_CHAIN = 1
       NUM_PORT_A = 2
       NUM_PORT_B = 2
       NUM_PORT_C = 1
       WIDTH_PORT_A = 64
       WIDTH_PORT_B = 128
       WIDTH_PORT_C = 64
       PLIOA_IDX2COL[0:1] = [0, 1]
       PLIOB_IDX2COL[0:1] = [0, 1]
       PLIOC_IDX2COL[0] = 1
    elif corenum == 16:
       NUM_CORE = 16
       assert(NUM_BATCH <= 24), "16 core per batch, max batch num is 24"
       NUM_CHAIN = 8
       NUM_PORT_A = 8
       NUM_PORT_B = 4
       NUM_PORT_C = 8
       WIDTH_PORT_A = 64
       WIDTH_PORT_B = 128
       WIDTH_PORT_C = 64
       PLIOA_IDX2COL[0:7] = [ 0,  1, 0, 1, 0, 1, 0, 1]
       PLIOB_IDX2COL[0:3] = [ 0,  1,  0,  1]
       PLIOC_IDX2COL[0:7] = [1, 0, 1, 0, 1, 0, 1, 0]
    elif corenum == 32:
       NUM_CORE = 32
       assert(NUM_BATCH <= 12), "32 core per batch, max batch num is 12"
       NUM_CHAIN = 16
       NUM_PORT_A = 8
       NUM_PORT_B = 8
       NUM_PORT_C = 16
       WIDTH_PORT_A = 128
       WIDTH_PORT_B = 128
       WIDTH_PORT_C = 64
       PLIOA_IDX2COL[0:7] = [ 2, 3, 2, 3, 0, 1, 0, 1]
       PLIOB_IDX2COL[0:7] = [ 0, 1, 0, 1, 2, 3, 2, 3]
       PLIOC_IDX2COL[0:15] = [1, 0, 1, 0, 1, 0, 1, 0, 3, 2, 3, 2, 3, 2, 3, 2]
    elif corenum == 64:
       NUM_CORE = 64
       assert(NUM_BATCH <= 6), "64 core per batch, max batch num is 6"
    else:
        print("[ERROR] Not Support this CORE NUM graph now!!!")
        sys.exit()
    col_end = BATCH0_COL_OFFSET + NUM_BATCH*(NUM_PORT_B//2) + (NUM_BATCH-1)*COL_INTERVAL
    assert(col_end < 50), "batch col idx exceed 49(COL_MAX)"

def showConfig():
	cfg = ''
	cfg += '-------------- summary of graph_conv ---------------\n'
	cfg += 'commit id: '+getCommitId() + "\n"
	form = '{:15}\t{:15}\n'
	cfg += form.format('Cores per Batch(CPB):', str(NUM_CORE))
	cfg += form.format('Batch Number(BAT_NUM):', str(NUM_BATCH))
	cfg += '----------------------------------------------------\n'
	cfg += 'Ports per Batch >> \n'
	form = '{:6}\t{:6}\t{:6}\t{:6}\n'
	cfg += form.format('type', 'prefix', 'number', 'width(bits)')
	cfg += form.format('ifm', 'aPr_', NUM_PORT_A, WIDTH_PORT_A)
	cfg += form.format('weights', 'bPr_', NUM_PORT_B, WIDTH_PORT_B)
	cfg += form.format('\t', 'batch share weights', MAX_BAT_PER_SHAREWGT, 1)
	cfg += form.format('ofm', 'cPr_', NUM_PORT_C, WIDTH_PORT_C)
	cfg += '----------------------------------------------------\n'
	cfg += 'Kernel >> \n'
	form = '{:30}\t{:35}\n'
	cfg += form.format('ifm->ptr', '<' + str(BUFA_PING_OFFSET) + ' ,' + str(BUFA_PONG_OFFSET) + '>')
	cfg += form.format('stack offset', str(STACK_OFFSET))
	cfg += form.format('layer_param_buf max size(word)', str(LAYER_PARAM_BUF_SIZE))
	cfg += form.format('wlut_buf max size(word)', str(WLUT_BUF_SIZE))
	cfg += '----------------------------------------------------\n'
	cfg += 'Options >> \n'
	return cfg
'''
	cfg += '----------------------------------------------------\n'
	cfg += 'RegMaps >>  saved at CONV_ONE_VEC array\nbelow offset is to CONV_ONE_VEC[0]\n'
	form = '{:8}\t{:8}\t{:8}\n'
	cfg += form.format('offset', 'size(bits)', 'contents')
	cfg += form.format('0x20', '32', 'commit id')
	cfg += form.format('0x24', '16', 'Cores per Batch')
	cfg += form.format('0x26', '16', 'Batch Number')
	cfg += form.format('0x28', '16', 'CONV_LEAKYRELU_EN')
'''


def checkJson(file):
    try:
        json.loads(file)
    except Exception as e:
        return e
    return True

def main(argv):
    if(len(argv) != 4):
        print("usage: ./gen_graph.py <core_num/batch> <batch num> <vver>")
        sys.exit()
    corenum = int(argv[1])
    batchnum = int(argv[2])
    vver = int(argv[3])
    assert(corenum*batchnum <= 400), "core_num * batch_num need <= 400"
    print ("[INFO] Generating Config...")
    genConfig(corenum, batchnum)
    print ("[INFO] Generating " + PARAM_FILE)
    genParamHPP()
    print ("[INFO] Generating " + GRAPH_FILE)
    genGraph()
    print ("[INFO] Generating " + PLATFORM_FILE)
    genPlatform()
    print ("[INFO] Generating " + PORTCONSTRAINT_FILE)
    genPortConstraint()
    portConstr = open(PORTCONSTRAINT_FILE).read()
    r = checkJson(portConstr)
    if r != True:
        print ("[ERROR]: " + PORTCONSTRAINT_FILE + " format error")
        print (r)
        sys.exit()
    print ("[INFO] Generated "+ argv[1] + " cores graph successful!!!")
    cfg = showConfig()
    fg = open("config.txt", "w")
    fg.write(cfg)
    fg.close()


if __name__ == "__main__":
	sys.exit(main(sys.argv))

# /*******************************************************************************
# /*                                                                         
# * Copyright 2019 Xilinx Inc.                                               
# *                                                                          
# * Licensed under the Apache License, Version 2.0 (the "License");          
# * you may not use this file except in compliance with the License.         
# * You may obtain a copy of the License at                                  
# *                                                                          
# *    http://www.apache.org/licenses/LICENSE-2.0                            
# *                                                                          
# * Unless required by applicable law or agreed to in writing, software      
# * distributed under the License is distributed on an "AS IS" BASIS,        
# * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. 
# * See the License for the specific language governing permissions and      
# * limitations under the License.                                           
# */
# *******************************************************************************/
