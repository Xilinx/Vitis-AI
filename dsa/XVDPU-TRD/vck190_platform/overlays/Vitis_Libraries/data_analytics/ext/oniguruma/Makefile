PYONY: all

MK_PATH := $(abspath $(lastword $(MAKEFILE_LIST)))
CUR_DIR := $(patsubst %/,%,$(dir $(MK_PATH)))
DEST_DIR ?= $(CUR_DIR)

all: 
	make -C $(CUR_DIR) lib

oniguruma:
	git clone https://github.com/kkos/oniguruma.git oniguruma

lib: oniguruma
	cd oniguruma && sh ./autogen.sh
	cd oniguruma && ./configure --prefix=$(CUR_DIR)/lib CFLAGS="-O3 -gdwarf-2 -g3"
	cd oniguruma && make
	cd oniguruma && make install

clean:
	rm -fr $(DEST_DIR)/oniguruma $(DEST_DIR)/lib

