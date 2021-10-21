#!/use/bin/env python3
from __future__ import print_function # this is solely to trap python2 without it the syntax error checks prevent execution

# argument checking and help
import argparse
parser = argparse.ArgumentParser(
    description="Example of the Heston Equation Finite Difference financial model running on a FPGA.\n\
Variables as detailed below are passed in so to run type \n\
'./fd_heston_test.py 1.5 0.04 0.3 -0.9 0.025 1 100 200 1 200 128 64 fd_heston,xclbin u250' \n\
Build instructions for the model hardware bitstream (xclbin) are in \n\
'/<path to xf_fintech>/L2/tests/FdEuropeanHestonEngine'",formatter_class=argparse.RawTextHelpFormatter)
parser.add_argument('kappa', type=float, help='mean reversion rate')
parser.add_argument('eta', type=float, help='long run average price')
parser.add_argument('sigma', type=float, help='volatility of volatility')
parser.add_argument('rho', type=float, help='correlation coefficient')
parser.add_argument('rd', type=float, help='risk free domestic interest rate')
parser.add_argument('T', type=float, help='expiration time')
parser.add_argument('K', type=float, help='strike price')
parser.add_argument('S', type=float, help='stock price')
parser.add_argument('V', type=float, help='volatility')
parser.add_argument('N', type=int, help='number of time steps')
parser.add_argument('m1', type=int, help='grid size for the Sdirection - m1')
parser.add_argument('m2', type=int, help='grid size for the Sdirectioni - m2')
parser.add_argument('load', type=str, help='filename of xlcbin load, e.g. fd_heston.xclbin')
parser.add_argument('card', type=str, help='type of the card - currently u200 or u250')
args = parser.parse_args()
# State financial model and args entered
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
from xf_fintech_python import DeviceManager, FDHeston

# Basic checking that the card and load arguments are correct
if not (args.card == "u250" or args.card == "u200"):
    sys.exit("This version executes on either card type u200 or u250")
if not exists(args.load):
    print("Bitstream load specified is ",args.load)
    sys.exit("Please check the supplied FPGA load filename - program does not see it")


# Declaring Variables
deviceList = DeviceManager.getDeviceList(args.card) # Pass in the card type from the command line

if len(deviceList) == 0 : # Check at least one card found
    sys.exit(("Please check that you have a "+args.card+" card present and ready for use"))

sGridOutput = []
vGridOutput = []
priceGridOutput = []

lastruntime = 0


# In this example financial data to test the module is fed in via the command line. Below is an example
# ./heston_fd_test.py 1.5 0.04 0.3 -0.9 0.025 1 100 200 1 200 128 64 fd_heston,xclbin u200
# and the expected value output for NPV is 111.832977

#Take each suffixed value and give it a meaningful name and convert from arv string format to numerical
meanReversionRate_kappa = args.kappa
longRunAveragePrice_eta = args.eta
volatilityOfVolatility_sigma = args.sigma
correlationCoefficient_rho = args.rho
riskFreeDomesticInterestRate_rd = args.rd
expirationtime_T = args.T
strikePrice_K = args.K
stockPrice_S = args.S
volatility_V = args.V
numberOfTimesteps_N = args.N
gridSizeForTheSdirection_m1 = args.m1
gridSizeForTheSdirection_m2 = args.m2
#Repeat back those values, the last three are integers.
print("So meanReversionRate_kappa is ",type(meanReversionRate_kappa)," and value is ",meanReversionRate_kappa)
print("and longRunAveragePrice_eta is ",type(longRunAveragePrice_eta)," and value is ",longRunAveragePrice_eta)
print("and volatilityOfVolatility_sigma is ",type(volatilityOfVolatility_sigma)," and value is ",volatilityOfVolatility_sigma)
print("and correlationCoefficient_rho is ",type(correlationCoefficient_rho)," and value is ",correlationCoefficient_rho)
print("and riskFreeDomesticInterestRate_rd is ",type(riskFreeDomesticInterestRate_rd)," and value is ",riskFreeDomesticInterestRate_rd)
print("and expirationtime_T is",type(expirationtime_T)," and value is ",expirationtime_T)
print("and strikePrice_K is ",type(strikePrice_K)," and value is ",strikePrice_K)
print("and stockPrice_S is ",type(stockPrice_S)," and value is ",stockPrice_S)
print("and volatility_V is ",type(volatility_V)," and value is ",volatility_V)
print("and numberOfTimesteps_N is ",type(numberOfTimesteps_N)," and value is ",numberOfTimesteps_N)
print("and gridSizeForTheSdirection_m1 is ",type(gridSizeForTheSdirection_m1)," and value is ",gridSizeForTheSdirection_m1)
print("and gridSizeForTheSdirection_m2 is ",type(gridSizeForTheSdirection_m2)," and value is ",gridSizeForTheSdirection_m2)
# Variables set and arrays for results, where necessary set


# Identify which cards are installed and choose the first available U250 card
print("Found these {0} device(s):".format(len(deviceList)))
for x in deviceList:
    print(x.getName())
print("Choosing the first suitable card\n")
chosenDevice = deviceList[0]

# Selecting and loading into FPGA on chosen card the financial model to be used
hestonFD = FDHeston(gridSizeForTheSdirection_m1, gridSizeForTheSdirection_m2, args.load)
hestonFD.claimDevice(chosenDevice)


#execute the calculation
print("\nRunning with number of steps specified,duplicating the c++ demonstration")
result = hestonFD.run(stockPrice_S, strikePrice_K, riskFreeDomesticInterestRate_rd, volatility_V, expirationtime_T, meanReversionRate_kappa, volatilityOfVolatility_sigma, correlationCoefficient_rho, longRunAveragePrice_eta, numberOfTimesteps_N )
print("\nSo the single output NPV is returning ",str(result))

runtime = hestonFD.lastruntime()
print("This run took", str(runtime), "microseconds") # display how long the processing took


print("\nNow running without specifying a number of steps")
result = hestonFD.run(stockPrice_S, strikePrice_K, riskFreeDomesticInterestRate_rd, volatility_V, expirationtime_T, meanReversionRate_kappa, volatilityOfVolatility_sigma, correlationCoefficient_rho, longRunAveragePrice_eta)
print("\nSo the single output NPV is returning ",str(result))

runtime = hestonFD.lastruntime()
print("This run took", str(runtime), "microseconds")


print("\nNow running with number of steps and feeding in dimensions of sgrid and vgrid, expecting arrays of results")
result = hestonFD.run(stockPrice_S, strikePrice_K, riskFreeDomesticInterestRate_rd, volatility_V, expirationtime_T, meanReversionRate_kappa, volatilityOfVolatility_sigma, correlationCoefficient_rho, longRunAveragePrice_eta, numberOfTimesteps_N, gridSizeForTheSdirection_m1, gridSizeForTheSdirection_m2, sGridOutput, vGridOutput, priceGridOutput)
print("\nSo the arrayed output is returning \n") # As there are a lot have gone with csv format output in the example
print("\nsGridvalues are: ")
for i in range(len(sGridOutput)):
                print(sGridOutput[i], end = ',')
                i += 1
print("\nvGridvalues are: ")
for i in range(len(vGridOutput)):
                print(vGridOutput[i], end = ',')
                i += 1
print("\npriceGridvalues are: ")
for i in range(len(priceGridOutput)):
                print(priceGridOutput[i], end = ',')
                i += 1

runtime = hestonFD.lastruntime()
print("\nThis run took", str(runtime), "microseconds")


#Relinquish ownership of the card
hestonFD.releaseDevice()
