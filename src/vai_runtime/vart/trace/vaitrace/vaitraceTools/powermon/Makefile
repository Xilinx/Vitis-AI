# Copyright 2022-2023 Advanced Micro Devices Inc.

PROJECT_NAME=libpowermon
TARGET=$(PROJECT_NAME).so


CC=$(CROSS_COMPILE)g++

CFLAGS= -share -fPIC -I./include -lstdc++

RM=rm -rf $(1)
CP=cp $(1) $(2)

SRCS = power_mon.cpp


all:
	$(CC) $(CFLAGS) $(SRCS) -O $(TARGET) 
ifneq($(OUT),)
	$(call CP,$(TARGET),$(OUT))
endif

clean:
	$(call RM,*.o)
	$(call RM,*.so)
