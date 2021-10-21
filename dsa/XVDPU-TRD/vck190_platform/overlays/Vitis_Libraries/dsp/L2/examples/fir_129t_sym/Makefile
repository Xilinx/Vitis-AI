#
# Copyright 2021 Xilinx, Inc.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
###############################################################################
#  Makefile used for AIE DSPLIB fir_sr_sym compilation and simulation
###############################################################################


ifeq ($(DSPLIB_ROOT), "")
	DSPLIB_REPO 	  := $(PWD)/../../../..
else
	DSPLIB_REPO 	  := $(DSPLIB_ROOT)
endif
# Includes
KERNEL_HDR         := $(DSPLIB_REPO)/L1/include/hw
KERNEL_SRC         := $(DSPLIB_REPO)/L1/src/hw
GRAPH_SRC          := $(DSPLIB_REPO)/L2/include/hw

NUM_SEC            := 1

NITER               = 8
LOCK_FILE          := .lock
INPUT_FILE         := ./data/input.txt

# Env config
STATUS_FILE        = status.txt
DIFF_OUT_FILE      := diff_output.txt
UUT_DATA_FILE      = ./data/uut_out.txt
REF_DATA_FILE      = ./data/fir_out_ref.txt
UUT_OUT_DIR        = ./aiesimulator_output
UUT_SIM_FILE       = data/output.txt
UUT_OUT_FILE       = $(UUT_OUT_DIR)/$(UUT_SIM_FILE)
DEVICE_FILE        = 4x4
TEST_BENCH         = test.cpp
WORK_DIR           = Work
LOG_FILE           = ./log.txt
PSLINKER_ARGS      = "-L xxx"
OTHER_OPTS         = --pl-freq=1000
SIM_OPTS           = --profile
PHRASE_1           = Compilation\ Complete
PHRASE_2           = Sim\ result\:\ 0
PHRASE_3           = identical

# Default make all
all          : clean create compile slp sim check_op get_status summary


create:
	@echo Start testing|& tee $(LOG_FILE);\
	 echo diff result > $(DIFF_OUT_FILE);\
	 echo "Results:" > $(STATUS_FILE);\
	rm -f $(LOCK_FILE)

compile:
	@echo COMPILE_START |& tee $(LOG_FILE);\
	date +%s |& tee -a $(LOG_FILE);\
	date |& tee -a $(LOG_FILE);\
	aiecompiler --dataflow -v --use-phy-shim=false -include=$(KERNEL_HDR) -include=$(KERNEL_SRC) -include=$(GRAPH_SRC) $(TEST_BENCH) $(OTHER_OPTS) --device=$(DEVICE_FILE) --test-iterations=$(NITER) -workdir=$(WORK_DIR) -Xpreproc=$(UUT_PREPROC_ARGS)|& tee -a $(LOG_FILE);\
	echo COMPILE_END |& tee -a $(LOG_FILE);\
	date +%s |& tee -a $(LOG_FILE)

sim:
	@echo SIM_START |& tee -a $(LOG_FILE);\
	date +%s |& tee -a $(LOG_FILE);\
	aiesimulator --pkg-dir=$(WORK_DIR) $(SIM_OPTS) |& tee -a $(LOG_FILE)
	echo SIM_END |& tee -a $(LOG_FILE);\
	date +%s |& tee -a $(LOG_FILE)

slp:
	@echo sleep for $(NUM_SEC) |& tee -a $(LOG_FILE);\
	sleep $(NUM_SEC)

check_op:
	@grep -ve '[XT]' $(UUT_OUT_FILE) > $(UUT_DATA_FILE);\
	echo "DIFF_START" >> $(LOG_FILE)
	diff -bc -s $(UUT_DATA_FILE) $(REF_DATA_FILE) >$(DIFF_OUT_FILE) || echo "DIFF_ERROR. Files don't match." >> $(LOG_FILE)
	echo "DIFF_END" >> $(LOG_FILE)

get_status:
	@grepres1=`grep  $(PHRASE_1) -c $(LOG_FILE)`;\
	 echo "    COMPILE:" $$grepres1 >>$(STATUS_FILE)
	@grepres2=`grep  $(PHRASE_2) -c $(LOG_FILE)`;\
	 echo "    SIM:" $$grepres2 >>$(STATUS_FILE)
	@grepres3=`grep  $(PHRASE_3) -c $(DIFF_OUT_FILE)`;\
	 echo "    FUNC:" $$grepres3 >>$(STATUS_FILE);\
	nl=`wc -l '$(UUT_DATA_FILE)' | grep -Eo '[0-9]+' |head -1`;\
	echo "    NUM_UUT_OUTPUTS:" $$nl >>$(STATUS_FILE);\
	t0=`grep COMPILE_START -A 1 $(LOG_FILE)|tail -n 1`;\
	t1=`grep COMPILE_END -A 1 $(LOG_FILE)|tail -n 1`;\
	td=`echo $$t1 - $$t0 |bc`;\
	echo "    COMPILE_TIME:" $$td >>$(STATUS_FILE);\
	t0=`grep SIM_START -A 1 $(LOG_FILE)|tail -n 1`;\
	t1=`grep SIM_END -A 1 $(LOG_FILE)|tail -n 1`;\
	td=`echo $$t1 - $$t0 |bc`;\
	echo "    SIM_TIME:" $$td >>$(STATUS_FILE);\
	echo test_complete > $(LOCK_FILE)

summary:
	cat $(STATUS_FILE)

.PHONY: clean
clean:
	@rm -rf aiesimulator_output
	rm -rf $(LOG_FILE)
	rm -rf  $(STATUS_FILE) $(DIFF_OUT_FILE) aiesimulator.log
	rm -f -R $(WORK_DIR) $(UUT_OUT_DIR)
	clear

