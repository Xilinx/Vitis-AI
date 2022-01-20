#! /bin/csh
#==============================================
DIR_PRJ         = $(shell pwd)
SIM_DIR         = $(DIR_PRJ)/workspace
RLS_DIR         = $(DIR_PRJ)/graph_conv_release
#----------------------------------------------

#config options
aie_activations_en=1
#-pl-freq = 125,250
PL_FREQ=125
#batch_0 start col
b0_col=6
#interval col between two batch
bat_interval=0
#number of batch share one weights port
bat_sharewgt=1
#chip:es1 es2 prod
CHIP=PROD

python_ver = python3
vver = 212
setenv_script = set_env.csh.212
compile_ip_script = compile_ip.csh.212

ifeq ($(ip_dir),)
	IP_DIR = $(DIR_PRJ)/ip
else
	IP_DIR = $(DIR_PRJ)/$(ip_dir)
endif

ifeq ($(pl_freq), 250)
	PL_FREQ = 250
endif

help:
	@echo 'make ip CPB=<32|64> BAT_NUM=<1~6> [ip_dir=./ip] pl_freq=<125|250> [aie_activations_en=$(aie_activations_en)]'

ip:
ifeq ($(strip $(CPB)),$(filter $(CPB),16 32 64))
	test -d $(IP_DIR) || mkdir -p $(IP_DIR)
	@echo  '[Compiling IP]:'
	@echo  '[Compiling IP]: IP DIR is :' $(IP_DIR) 
	@echo  '[Compiling IP]: [step 1]. coping scripts'
	cp ./scripts/${setenv_script} ${IP_DIR}/set_env.csh
	@echo 'setenv DIR_PRJ '${DIR_PRJ} >> ${IP_DIR}/set_env.csh
	@echo 'setenv IP_DIR '${IP_DIR} >> ${IP_DIR}/set_env.csh
	cp ./scripts/${compile_ip_script} ${IP_DIR}/compile_ip.csh
	sed -i "s/-pl-freq=[0-9]\+/-pl-freq=${PL_FREQ}/g" ${IP_DIR}/compile_ip.csh
	@echo  '[Compiling IP]: [step 2]. generating IP graph'
	sed -i "s/PL_FREQ=[0-9]\+/PL_FREQ=$(PL_FREQ)/g" ./gen_graph.py
	sed -i "s/VVER=[0-9]\+/VVER=$(vver)/g" ./gen_graph.py
	sed -i "s/BATCH0_COL_OFFSET=[0-9]\+/BATCH0_COL_OFFSET=$(b0_col)/g" ./gen_graph.py
	sed -i "s/COL_INTERVAL=[0-9]\+/COL_INTERVAL=$(bat_interval)/g" ./gen_graph.py
	sed -i "s/BAT_SHAREWGT=[0-9]\+/BAT_SHAREWGT=$(bat_sharewgt)/g" ./gen_graph.py
	${python_ver} gen_graph.py $(CPB) $(BAT_NUM) $(vver)
	@echo "AIE_ACTIVATIONS_EN  ${aie_activations_en}" >> config.txt
	@echo "CHIP $(CHIP)" >> config.txt
	@echo "PL_FREQ ${PL_FREQ}" >> config.txt
	sed -i "s/AIE_ACTIVATIONS_EN [0-1]/AIE_ACTIVATIONS_EN $(aie_activations_en)/g" ./src/param.h
	sed -i "s/CHIP PROD/CHIP $(CHIP)/g" ./src/param.h
	sed -i "s/VVER [0-9]\+/VVER $(vver)/g" ./src/param.h
	mv config.txt ${IP_DIR}
	@echo  '[Compiling IP]: [step 3]. run aiecompiler'
	@echo ""> src/test_param.h
	cd $(IP_DIR) && ./compile_ip.csh	
	cp ./scripts/getAddr.py $(IP_DIR) && cd $(IP_DIR) && ${python_ver} getAddr.py 0 $(DIR_PRJ)/src/graph_conv.h >> config.txt
	@echo  '[Compiling IP]: IP generated at '$(IP_DIR)'/Work'
	@echo "aiecompiler version: ${vver}" >> $(IP_DIR)/config.txt
	cat $(IP_DIR)/config.txt
else
	@echo "ERROR: Not Supported CPB(Cores per batch)!!!" ;
endif

clean_ip:
	rm -rf ./ip
clean:
	rm -rf ./workspace
