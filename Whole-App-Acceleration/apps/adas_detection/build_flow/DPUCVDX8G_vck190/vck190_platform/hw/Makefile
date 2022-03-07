VIVADO = $(XILINX_VIVADO)/bin/vivado
TCL_SOURCE=scripts/main.tcl

all: $(XSA)

$(XSA): $(TCL_SOURCE)
	$(VIVADO) -mode batch -notrace -source $(TCL_SOURCE) -tclargs $(PLATFORM_NAME) $(VER)

clean:
	$(RM) -r vivado* build .Xil *dynamic* *.log *.xpe project* *.xsa 
