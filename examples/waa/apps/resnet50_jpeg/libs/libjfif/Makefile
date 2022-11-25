DEBUG=1

VPATH=./src/:./examples
SLIB=build/libjfif.so
ALIB=build/libjfif.a
EXEC=jfif
OBJDIR=./obj/

#SYSROOT?= <path for SYSROOT>
															 
#CC=gcc
AR=ar
ARFLAGS=rcs
OPTS=-Ofast
LDFLAGS= -lm -pthread 
#COMMON= -Iinclude/ -Isrc/ 
COMMON= -Iinclude/ -Isrc/ -I$(SYSROOT)/usr/include
CFLAGS=-Wall -Wno-unknown-pragmas -Wfatal-errors -fPIC

ifeq ($(DEBUG), 1) 
OPTS=-O0 -g
endif

CFLAGS+=$(OPTS)
#CC:= aarch64-linux-gnu-gcc

OBJ=tables.o parser.o composer.o jfif.o
EXECOBJA=main.o

EXECOBJ = $(addprefix $(OBJDIR), $(EXECOBJA))
OBJS = $(addprefix $(OBJDIR), $(OBJ))
DEPS = $(wildcard src/*.h) Makefile include/jfif.h

all: obj build $(SLIB) $(ALIB) $(EXEC)


$(EXEC): $(EXECOBJ) $(ALIB)
	$(CC) $(COMMON) $(CFLAGS) $^ -o $@ $(LDFLAGS) $(ALIB)

$(ALIB): $(OBJS)
	$(AR) $(ARFLAGS) $@ $^

$(SLIB): $(OBJS)
	$(CC) $(CFLAGS) -shared $^ -o $@ $(LDFLAGS)

$(OBJDIR)%.o: %.c $(DEPS)
	$(CC) $(COMMON) $(CFLAGS) -c $< -o $@

obj:
	mkdir -p obj
build:
	mkdir -p build

.PHONY: clean

clean:
	rm -rf $(OBJS) $(SLIB) $(ALIB) $(EXEC) $(EXECOBJ)
	rm -rf obj build scan_0.bin scan_1.bin scan_2.bin output.jpg
