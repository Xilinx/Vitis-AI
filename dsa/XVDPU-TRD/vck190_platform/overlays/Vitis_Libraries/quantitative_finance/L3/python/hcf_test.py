#!/usr/bin/env python3

# argument checking and help
import argparse
parser = argparse.ArgumentParser(
    description='Example of the Heston Closed Form financial model running on a FPGA.')
required = parser.add_argument_group("required arguments")
required.add_argument("-x","--xclbin_file", dest="xclbin_filename", type=str, required=True,
    help="The model hardware bitstream xclbin filename. Build instructions for which are in '/<path to xf_fintech>/L2/tests/HCFEngine'")
required.add_argument("-c","--card",dest="card_type", type=str,required=True,
    help='Current supported Alveo cards are u200 and u250')
args = parser.parse_args()
# State test financial model and args entered
print("+--------------------------------------------------------------------")
print(parser.description)
print(args)
print("+--------------------------------------------------------------------")

# Ensure environmental variables i.e. paths are set to used the modules
import sys
# Check not using python 2
if sys.version.startswith("2"):
    sys.exit("Seem to be running with the no longer supported python 2 - require version 3")
from os.path import exists
from xf_fintech_python import DeviceManager, hcf_input_data,hcf 

# Basic checking that the number of arguments are correct
if not (args.card_type == "u250" or args.card_type == "u200"):
    sys.exit("This version executes on either card type u200 or u250")
if not exists(args.xclbin_filename):
    sys.exit("Please check the supplied FPGA load filename - program does not see it")

# Declaring Variables

# order is StockPrice, StrikePrice, RiskFreeInterestRate,Volatility,time to vest (years),
#          kappa the expected[Y-1] where Y is the random variable, lambda - mean jump per unit time,
#          lambda - mean jump per unit time, and finally the expected result

inputDataList =[]    # This is your list of input data , a list of lists
outputList = [] # This will be filled with a list of OptionPrices, one for each row of data input
numberOptions = 16
#Model Input Data
s0 = 80.0   # stock price at t=0
v0 = 0.1    # stock price variance at t=0
K = 100.0   # strike price
rho = -0.9  # correlation of the 2 Weiner processes
T = 1.0     # expiration time
r = 0.05    # risk free interest rate
vvol = 0.3  # volatility of volatility (sigma)
vbar = 0.04 # long term average variance (theta)
kappa = 1.5 # rate of reversion

# populate some data
for loop in range(0, numberOptions) :
    inputDataList.append(hcf_input_data())
    inputDataList[loop].s0 = s0 + (3 * loop)
    inputDataList[loop].v0= v0
    inputDataList[loop].K = K
    inputDataList[loop].rho = rho
    inputDataList[loop].T = T
    inputDataList[loop].r = r
    inputDataList[loop].kappa = kappa
    inputDataList[loop].vvol = vvol
    inputDataList[loop].vbar= vbar

print("[XF_FINTECH] ==========");
print("[XF_FINTECH] Parameters");
print("[XF_FINTECH] ==========");
print("[XF_FINTECH] Strike price                       = ", K);
print("[XF_FINTECH] Rho (Weiner process correlation)   = ", rho);
print("[XF_FINTECH] Time to maturity                   = ", T);
print("[XF_FINTECH] Risk free interest rate            = ", r);
print("[XF_FINTECH] Rate of reversion (kappa)          = ", kappa);
print("[XF_FINTECH] volatility of volatility (sigma)   = ", vvol);
print("[XF_FINTECH] Long term average variance (theta) = ", vbar);


deviceList = DeviceManager.getDeviceList(args.card_type) # Pass in the card type from the command line

if len(deviceList) == 0 : # Check at least one card found
    sys.exit(("Please check that you have a "+args.card_type+" card present and ready for use"))

print("Found these {0} device(s):".format(len(deviceList)))
for x in deviceList:
    print(x.getName())

print("Choosing the first suitable card\n")
chosenDevice = deviceList[0]


hestonCF= hcf(args.xclbin_filename)
hestonCF.claimDevice(chosenDevice)

hestonCF.run(inputDataList, outputList, numberOptions) #This is the call to process contents of dataList

#Format output to match the example in C++, simply to aid comparison of results
print("[XF_FINTECH] Multiple [",numberOptions,"] Options European Call");
for loop in range(0, numberOptions) :
    print("[XF_FINTECH] Option %2d"%loop,"\tOptionPrice = %8.5f"%outputList[loop])

print("\nEnd of example.")

hestonCF.releaseDevice()

sys.exit(0)
