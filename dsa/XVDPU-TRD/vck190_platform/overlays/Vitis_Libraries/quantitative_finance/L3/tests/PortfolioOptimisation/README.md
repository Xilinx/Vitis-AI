## Portfolio Optimisation L3 Test 
This is a test of the Portfolio Optimisation model using the Vitis environment.  It supports software and hardware emulation as well as running the hardware accelerator on supported Alveo cards.

It uses a fixed set of test data with expected values as calculated by the C Model the kernel was developed from. The test will take the input data and calculate asset weights, portfolio expected return and portfolio variance for 4 different types of portfolio; the Global Miminum Variance Portfolio, an Efficient Portfolio of risky assets for a given target return, the Tangency Portfolio for a given risk free rate and an Efficient Portfolio composed of risky assets and a risk free asset for given target return and risk free rate.

## Prerequisites
- Alveo card (eg U200) installed and configured as per https://www.xilinx.com/products/boards-and-kits/alveo/u200.html#gettingStarted
- Xilinx runtime (XRT) installed
- Xilinx Vitis 2020.1 installed and configured

## Building the demonstration
The demonstration application and kernel is built using a command line Makefile flow.

### Step 1 :
Setup the build environment using the Vitis and XRT scripts:

            source <install path>/Vitis/2020.1/settings64.sh
            source /opt/xilinx/xrt/setup.sh

### Step 2 :
Call the Makefile. For example:

	make check TARGET=sw_emu DEVICE=xilinx_u200_xdma_201830_2

	make check TARGET=hw_emu DEVICE=xilinx_u200_xdma_201830_2

	make all TARGET=hw DEVICE=xilinx_u200_xdma_201830_2
        
	make run TARGET=hw DEVICE=xilinx_u200_xdma_201830_2

The Makefile supports software emulation, hardware emulation and hardware targets ('sw_emu', 'hw_emu' and 'hw', respectively).



If the make check or run option is used the demo will automatically be lauched.


## Example Output
Global Minimum Variance Portfolio

GMVP w[0] = 0.231147

GMVP w[1] = -0.00221761

GMVP w[2] = -0.084974

GMVP w[3] = 0.106045

GMVP w[4] = 0.00620003

GMVP w[5] = 0.411149

GMVP w[6] = -0.0693706

GMVP w[7] = -0.0359753

GMVP w[8] = 0.123788

GMVP w[9] = 0.314209

GMVP Expected return = 0.00900249

GMVP Variance = 0.000977613

Efficient Portfolio with target return of 0.02

Eff w[0] = 0.603973

Eff w[1] = -0.0421184

Eff w[2] = -0.0811392

Eff w[3] = -0.0440177

Eff w[4] = 0.207977

Eff w[5] = 0.247975

Eff w[6] = -0.0878477

Eff w[7] = -0.0300707

Eff w[8] = 0.0807078

Eff w[9] = 0.144561

Eff Expected return = 0.02

Eff Variance = 0.0014647

Tangency Portfolio for risk free rate of 0.001

Tan w[0] = 1.25948

Tan w[1] = -0.112273

Tan w[2] = -0.0743967

Tan w[3] = -0.307862

Tan w[4] = 0.562747

Tan w[5] = -0.0389208

Tan w[6] = -0.120334

Tan w[7] = -0.0196891

Tan w[8] = 0.00496252

Tan w[9] = -0.153718

Tan Expected return = 0.0393361

Tan Variance = 0.00468328

Tan Sharpe Ratio = 0.560187

Tangency Portfolio for risk free rate of 0.001 and target return 0.02

Eff Tan w[0] = 0.624221

Eff Tan w[1] = -0.0556444

Eff Tan w[2] = -0.0368722

Eff Tan w[3] = -0.152581

Eff Tan w[4] = 0.278906

Eff Tan w[5] = -0.0192898

Eff Tan w[6] = -0.0596396

Eff Tan w[7] = -0.00975822

Eff Tan w[8] = 0.0024595

Eff Tan w[9] = -0.0761853

Eff Tan proportion in risk free = 0.504384

xecutionTime = 10255 microseconds)

Portfolio Optimisation releasing device...

TEST PASS

