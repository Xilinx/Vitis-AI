## Binomial Tree  Demonstration
This is a demonstration of the BinomialTree solver using the Vitis environment.  It supports software and hardware emulation as well as running the hardware accelerator on supported Alveo cards.

It uses a fixed set of test data produced from quantlib (Black Scholes for European & Binomial Leisen Reimer for American) stored in the data directory and the output from the host/kernel. The demo will take the input data, compute a number of options using the kernel and then compare this to the reference data.  The largest difference between the reference data and the kernel computed grid will be displayed as well as the root mean squared.

## Prerequisites
- Alveo card (eg U200) installed and configured as per https://www.xilinx.com/products/boards-and-kits/alveo/u200.html#gettingStarted
- Xilinx runtime (XRT) installed
- Xilinx Vitis 2020.1 installed and configured

## Building the Finite Difference demonstration
The demonstration application and kernel is built using a command line Makefile flow.

### Step 1 :
Setup the build environment using the Vitis and XRT scripts:

            source <install path>/Vitis/2020.1/settings64.sh
            source /opt/xilinx/xrt/setup.sh

### Step 2 :
Call the Makefile. For example:

	make run TARGET=sw_emu DEVICES=xilinx_u200_xdma_201830_2 DT=double PE=1

	make run TARGET=hw_emu DEVICES=xilinx_u200_xdma_201830_2 DT=double PE=1

	make all TARGET=hw DEVICES=xilinx_u200_xdma_201830_2 DT=double PE=1
        

The Makefile supports software emulation, hardware emulation and hardware targets ('sw_emu', 'hw_emu' and 'hw', respectively).  Additionally, several parameters are used to configure the host and/or kernel build. Most have restricted acceptable values and several must be consistent with others for the build to succeed.

| Makefile parameter | Default | Kernel/Host | Decription                                   | Valid Values                                |
|--------------------|---------|-------------|----------------------------------------------|---------------------------------------------|
|DT                  | double  | Kernel      | Datatype size			            | float,   double                             |
|PE                  | 1       | Kernel      | Number of parallel option calculations       | 1, 2, 4, 8                                  |



If the make run option is used the demo will automatically be lauched or else its possible to manually launch the exe.

For example to run with a prebuilt kernel you can use:
	    ./binarytree_demo


## Example Output for AmericanPut
##================================================================
##Testing:american_put
##================================================================

CRR_Bn_T001:  (Host) RMS:0.00000000000110 Max:0.00000000000567    RMS:0.00144519352882 Max:0.00611697737115

CRR_Bn_T002:  (Host) RMS:0.00000000000074 Max:0.00000000000365    RMS:0.00255336112672 Max:0.00834277026814

CRR_Bn_T003:  (Host) RMS:0.00000000000000 Max:0.00000000000000    RMS:0.00381593274716 Max:0.01360156107371

CRR_Bn_T004:  (Host) RMS:0.00000000000000 Max:0.00000000000000    RMS:0.00417088166233 Max:0.01844669995538

CRR_Bn_T005:  (Host) RMS:0.00000000000196 Max:0.00000000000526    RMS:0.00338581307482 Max:0.01205327842538

CRR_Bn_T006:  (Host) RMS:0.00000000000124 Max:0.00000000000653    RMS:0.00143836386492 Max:0.00607348061548

CRR_Bn_T007:  (Host) RMS:0.00000000000022 Max:0.00000000000156    RMS:0.00253433944175 Max:0.00833764487649

CRR_Bn_T008:  (Host) RMS:0.00000000000000 Max:0.00000000000001    RMS:0.00372165768290 Max:0.01322251282404

CRR_Bn_T009:  (Host) RMS:0.00000000000000 Max:0.00000000000001    RMS:0.00402738407767 Max:0.01660399278278

CRR_Bn_T010:  (Host) RMS:0.00000000000081 Max:0.00000000000270    RMS:0.00335018759472 Max:0.01052755473169

CRR_Bn_T011:  (Host) RMS:0.00000000000091 Max:0.00000000000448    RMS:0.00143080485994 Max:0.00601130598642

CRR_Bn_T012:  (Host) RMS:0.00000000000008 Max:0.00000000000053    RMS:0.00251363060479 Max:0.00833261535561

CRR_Bn_T013:  (Host) RMS:0.00000000000000 Max:0.00000000000002    RMS:0.00361428051447 Max:0.01270680261257

CRR_Bn_T014:  (Host) RMS:0.00000000000402 Max:0.00000000000824    RMS:0.00394455356389 Max:0.01418682733515

CRR_Bn_T015:  (Host) RMS:0.00000000000061 Max:0.00000000000210    RMS:0.00372751149893 Max:0.00868411747416

CRR_Bn_T016:  (Host) RMS:0.00000000000081 Max:0.00000000000375    RMS:0.00140356887467 Max:0.00570981730682

CRR_Bn_T017:  (Host) RMS:0.00000000000000 Max:0.00000000000001    RMS:0.00243949702471 Max:0.00828646780377

CRR_Bn_T018:  (Host) RMS:0.00000000000000 Max:0.00000000000001    RMS:0.00323860273029 Max:0.01067721966793

CRR_Bn_T019:  (Host) RMS:0.00000000000000 Max:0.00000000000001    RMS:0.00342813056906 Max:0.01032682495247

CRR_Bn_T020:  (Host) RMS:0.00000000000023 Max:0.00000000000082    RMS:0.00664408415374 Max:0.02971915551424

CRR_Bn_T021:  (Host) RMS:0.00000000000089 Max:0.00000000000416    RMS:0.00138024672942 Max:0.00567719087408

CRR_Bn_T022:  (Host) RMS:0.00000000000000 Max:0.00000000000000    RMS:0.00234778893574 Max:0.00820362568525

CRR_Bn_T023:  (Host) RMS:0.00000000000000 Max:0.00000000000002    RMS:0.00287277858118 Max:0.00868742573120

##================================================================
