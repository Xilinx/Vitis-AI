PROJECT_NAME=libnoc
TARGET=$(PROJECT_NAME).so

CC=$(CROSS_COMPILE)g++

CFLAGS = -shared -fPIC -I./include 

RM=rm -rf $(1)
CP=cp $(1) $(2)

SRCS = noc.cpp

all:
		$(CC) $(CFLAGS) $(SRCS) -o $(TARGET)
		ifneq ($(OUT),)
			$(call CP,$(TARGET),$(OUT))
		endif

clean:
		$(call RM,*.o)
			$(call RM,*.so)

