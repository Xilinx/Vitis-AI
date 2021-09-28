.PHONY: all clean

MK_PATH := $(abspath $(lastword $(MAKEFILE_LIST)))
CUR_DIR := $(patsubst %/,%,$(dir $(MK_PATH)))

all:
	flock $(CUR_DIR)/.lock make -C $(CUR_DIR) lib

lib: dcmt
	make -C $(CUR_DIR)/dcmt/lib

dcmt:
	git clone https://github.com/MersenneTwister-Lab/dcmt dcmt



clean:
	rm -rf dcmt 
