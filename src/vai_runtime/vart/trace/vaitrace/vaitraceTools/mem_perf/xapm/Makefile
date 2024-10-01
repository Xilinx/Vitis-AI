# Copyright 2022-2023 Advanced Micro Devices, Inc. All rights reserved.

PROJECT_NAME=libxapm
TARGET=$(PROJECT_NAME).so

CC=$(CROSS_COMPILE)g++

CFLAGS = -shared -fPIC -I./include -lstdc++

RM=rm -rf $(1)
CP=cp $(1) $(2)

SRCS = apm.cpp apm_shell.cpp

all:
	$(CC) $(CFLAGS) $(SRCS) -o $(TARGET)
ifneq ($(OUT),)
	$(call CP,$(TARGET),$(OUT))
endif

clean:
	$(call RM,*.o)
	$(call RM,*.so)

