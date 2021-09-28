.PHONY: all clean

MK_PATH := $(abspath $(lastword $(MAKEFILE_LIST)))
CUR_DIR := $(patsubst %/,%,$(dir $(MK_PATH)))
DEST_DIR ?= $(CUR_DIR)

SF ?= 1

all:
	flock $(DEST_DIR)/.lock make -C $(DEST_DIR) -f $(MK_PATH) sf$(SF)/lineitem.tbl

dbgen:
	git clone https://github.com/electrum/ssb-dbgen.git dbgen

dbgen/dbgen: | dbgen
	sed -i 's/MACHINE =MAC/MACHINE =LINUX/' dbgen/makefile
	sed -i 's/WORKLOAD =SSBM/WORKLOAD =TPCH/' dbgen/makefile
	make -C dbgen

sf%:
	mkdir -p sf$(*)

sf%/lineitem.tbl: | dbgen/dbgen sf%
	(cd dbgen && ./dbgen -s $(*) -f && mv *.tbl ../sf$(*))

clean:
	flock $(DEST_DIR)/.lock rm -rf $(DEST_DIR)/dbgen $(DEST_DIR)/sf*
