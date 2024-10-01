# Copyright 2022-2023 Advanced Micro Devices Inc.

CC=$(CROSS_COMPILE)g++

CFLAGS = -shared -O3 -fPIC -std=c++11

RM=rm -rf $(1)
CP=cp $(1) $(2)

all: $(subst .cpp,.so,$(wildcard *dpu*.cpp))

%.so: %.cpp inc/%.h
	$(CC) $(CFLAGS)  $< -o $@ 

clean:
	$(call RM,*.o)
	$(call RM,*.so)

