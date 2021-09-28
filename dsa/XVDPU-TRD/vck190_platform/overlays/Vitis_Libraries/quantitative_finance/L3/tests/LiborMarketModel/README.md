
# Setup Environment

source /opt/xilinx/xrt/setup.csh

source /*path to xf_fintech*/L3/src/env.csh

# Build Xilinx Fintech Library

cd  /*path to xf_fintech*/L3/src

**make clean all**


# Build Instuctions

To build the command line executable (lmm_cl) from this directory

**make clean all**

> Note this requires the xilinx fintech library to already to built


# Run Instuctions

Copy the prebuilt kernel files from /*path to xf_fintech*/L2/tests/LiborMarketModel/ to this directory

**lmm_cap.xclbin**
**lmm_ratchetfloater.xclbin**
**lmm_ratchetcap.xclbin**

To run the command line exe and test different pricers run

>output/lmm_cl.exe -l liborRates.csv -v capletVolas.csv [-b beta] [-p paths] [-n notional] [-c caprate] [-x rfX] [-y rfY] [-a rfAlpha] [-s spread] [-k kappa0]

or with default arguments

**make run**
