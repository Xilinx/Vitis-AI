SRC := ./
INC := ./
VPATH := $(SRC):$(INC)


bin_gen.so: bin_generator.cc
	g++ -I $(INC) -std=c++11 -shared -fPIC -o $@ $^
	
.PHONY: clean

clean:
	rm -rf *.so

