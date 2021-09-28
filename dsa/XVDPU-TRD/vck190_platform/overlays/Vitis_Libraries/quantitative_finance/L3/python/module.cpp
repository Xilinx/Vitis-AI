/*
 * Copyright 2019 Xilinx, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "pybind11/iostream.h"
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"

#include "xf_fintech_api.hpp"

using namespace xf::fintech;

namespace py = pybind11;

PYBIND11_MODULE(xf_fintech_python, m) {
    py::add_ostream_redirect(m, "OStreamRedirect"); // to redirect stdout/stderr to python sys:stdout and sys::stderr

    py::enum_<OptionType>(m, "OptionType")
        .value("Call", OptionType::Call)
        .value("Put", OptionType::Put)
        .export_values();

    py::class_<Device> device(m, "Device");
    device.def(py::init())
        .def(py::init<cl::Device>())
        .def("claim", &Device::claim)
        .def("release", &Device::release)
        .def("getName", &Device::getName)
        .def("getDeviceType", &Device::getDeviceType);

    py::enum_<Device::DeviceType>(device, "DeviceType")
        .value("U50", Device::DeviceType::U50)
        .value("U200", Device::DeviceType::U200)
        .value("U250", Device::DeviceType::U250)
        .value("U280", Device::DeviceType::U280)
        .export_values();

    py::class_<DeviceManager, std::unique_ptr<DeviceManager, py::nodelete> >(m, "DeviceManager")
        .def(py::init([]() { return std::unique_ptr<DeviceManager, py::nodelete>(DeviceManager::getInstance()); }))
        .def_static("getDeviceList", (std::vector<Device*>(*)(void)) & DeviceManager::getDeviceList,
                    py::return_value_policy::reference_internal, py::call_guard<py::scoped_ostream_redirect>())
        .def_static("getDeviceList", (std::vector<Device*>(*)(std::string)) & DeviceManager::getDeviceList,
                    py::return_value_policy::reference_internal, py::call_guard<py::scoped_ostream_redirect>())
        .def_static("getDeviceList", (std::vector<Device*>(*)(Device::DeviceType)) & DeviceManager::getDeviceList,
                    py::return_value_policy::reference_internal, py::call_guard<py::scoped_ostream_redirect>());

    py::class_<MCEuropean>(m, "MCEuropean")
        .def(py::init<std::string>())
        .def("claimDevice", &MCEuropean::claimDevice, py::call_guard<py::scoped_ostream_redirect>())
        .def("releaseDevice", &MCEuropean::releaseDevice, py::call_guard<py::scoped_ostream_redirect>())
        .def("deviceIsPrepared", &MCEuropean::deviceIsPrepared, py::call_guard<py::scoped_ostream_redirect>())
        .def("lastruntime", &MCEuropean::getLastRunTime)

        .def("run",
             [](MCEuropean& self, OptionType optionType, double stockPrice, double strikePrice, double riskFreeRate,
                double dividendYield, double volatility, double timeToMaturity, double requiredTolerance) {
                 int retval;
                 double optionPrice;

                 py::scoped_ostream_redirect outStream(std::cout, py::module::import("sys").attr("stdout"));

                 retval = self.run(optionType, stockPrice, strikePrice, riskFreeRate, dividendYield, volatility,
                                   timeToMaturity, requiredTolerance, &optionPrice);

                 return std::make_tuple(retval, optionPrice);
             })

        .def("run",
             [](MCEuropean& self, OptionType optionType, double stockPrice, double strikePrice, double riskFreeRate,
                double dividendYield, double volatility, double timeToMaturity, unsigned int requiredNumSamples) {
                 int retval;
                 double optionPrice;

                 py::scoped_ostream_redirect outStream(std::cout, py::module::import("sys").attr("stdout"));

                 retval = self.run(optionType, stockPrice, strikePrice, riskFreeRate, dividendYield, volatility,
                                   timeToMaturity, requiredNumSamples, &optionPrice);

                 return std::make_tuple(retval, optionPrice);
             })

        .def("run",
             [](MCEuropean& self, std::vector<OptionType> optionTypeList, std::vector<double> stockPriceList,
                std::vector<double> strikePriceList, std::vector<double> riskFreeRateList,
                std::vector<double> dividendYieldList, std::vector<double> volatilityList,
                std::vector<double> timeToMaturityList, std::vector<double> requiredToleranceList) {
                 int retval;
                 unsigned int numAssets = stockPriceList.size(); // use the length of the stock price list to determine
                                                                 // how many assets we are dealing with...
                 std::vector<double> optionPriceVector(numAssets);

                 py::scoped_ostream_redirect outStream(std::cout, py::module::import("sys").attr("stdout"));

                 retval = self.run(optionTypeList.data(), stockPriceList.data(), strikePriceList.data(),
                                   riskFreeRateList.data(), dividendYieldList.data(), volatilityList.data(),
                                   timeToMaturityList.data(), requiredToleranceList.data(), optionPriceVector.data(),
                                   numAssets);

                 return std::make_tuple(retval, optionPriceVector);
             })

        .def("run", [](MCEuropean& self, std::vector<OptionType> optionTypeList, std::vector<double> stockPriceList,
                       std::vector<double> strikePriceList, std::vector<double> riskFreeRateList,
                       std::vector<double> dividendYieldList, std::vector<double> volatilityList,
                       std::vector<double> timeToMaturityList, std::vector<unsigned int> requiredNumSamples,
                       py::list outputResults) {
            int retval;
            unsigned int numAssets = stockPriceList.size(); // use the length of the stock price list to determine how
                                                            // many assets we are dealing with...
            std::vector<double> optionPriceVector(numAssets);

            py::scoped_ostream_redirect outStream(std::cout, py::module::import("sys").attr("stdout"));

            retval =
                self.run(optionTypeList.data(), stockPriceList.data(), strikePriceList.data(), riskFreeRateList.data(),
                         dividendYieldList.data(), volatilityList.data(), timeToMaturityList.data(),
                         requiredNumSamples.data(), optionPriceVector.data(), numAssets);

            for (auto i : optionPriceVector) outputResults.append(i);

            return retval;
        });

    py::class_<MCAmerican>(m, "MCAmerican")
        .def(py::init())
        .def("claimDevice", &MCAmerican::claimDevice, py::call_guard<py::scoped_ostream_redirect>())
        .def("releaseDevice", &MCAmerican::releaseDevice, py::call_guard<py::scoped_ostream_redirect>())
        .def("deviceIsPrepared", &MCAmerican::deviceIsPrepared, py::call_guard<py::scoped_ostream_redirect>())
        .def("lastruntime", &MCAmerican::getLastRunTime)

        .def("run",
             [](MCAmerican& self, OptionType optionType, double stockPrice, double strikePrice, double riskFreeRate,
                double dividendYield, double volatility, double timeToMaturity, double requiredTolerance) {
                 int retval;
                 double optionPrice;

                 py::scoped_ostream_redirect outStream(std::cout, py::module::import("sys").attr("stdout"));

                 retval = self.run(optionType, stockPrice, strikePrice, riskFreeRate, dividendYield, volatility,
                                   timeToMaturity, requiredTolerance, &optionPrice);

                 return std::make_tuple(retval, optionPrice);
             })

        .def("run",
             [](MCAmerican& self, OptionType optionType, double stockPrice, double strikePrice, double riskFreeRate,
                double dividendYield, double volatility, double timeToMaturity, unsigned int requiredNumSamples) {
                 int retval;
                 double optionPrice;

                 py::scoped_ostream_redirect outStream(std::cout, py::module::import("sys").attr("stdout"));

                 retval = self.run(optionType, stockPrice, strikePrice, riskFreeRate, dividendYield, volatility,
                                   timeToMaturity, requiredNumSamples, &optionPrice);

                 return std::make_tuple(retval, optionPrice);
             });

    py::class_<MCEuropeanDJE>(m, "MCEuropeanDJE")
        .def(py::init<std::string>())
        .def("claimDevice", &MCEuropeanDJE::claimDevice, py::call_guard<py::scoped_ostream_redirect>())
        .def("releaseDevice", &MCEuropeanDJE::releaseDevice, py::call_guard<py::scoped_ostream_redirect>())
        .def("deviceIsPrepared", &MCEuropeanDJE::deviceIsPrepared, py::call_guard<py::scoped_ostream_redirect>())
        .def("lastruntime", &MCEuropeanDJE::getLastRunTime, py::call_guard<py::scoped_ostream_redirect>())

        .def("run",
             [](MCEuropeanDJE& self, std::vector<OptionType> optionTypeList, std::vector<double> stockPriceList,
                std::vector<double> strikePriceList, std::vector<double> riskFreeRateList,
                std::vector<double> dividendYieldList, std::vector<double> volatilityList,
                std::vector<double> timeToMaturityList, std::vector<double> requiredToleranceList, double dowDivisor) {
                 int retval;
                 unsigned int numAssets = stockPriceList.size(); // use the length of the stock price list to determine
                                                                 // how many assets we are dealing with...
                 std::vector<double> optionPriceVector(numAssets);
                 double DJIAoptionOutput;

                 py::scoped_ostream_redirect outStream(std::cout, py::module::import("sys").attr("stdout"));

                 retval = self.run(optionTypeList.data(), stockPriceList.data(), strikePriceList.data(),
                                   riskFreeRateList.data(), dividendYieldList.data(), volatilityList.data(),
                                   timeToMaturityList.data(), requiredToleranceList.data(), numAssets, dowDivisor,
                                   &DJIAoptionOutput);

                 return std::make_tuple(retval, DJIAoptionOutput);
             })

        .def("run", [](MCEuropeanDJE& self, std::vector<OptionType> optionTypeList, std::vector<double> stockPriceList,
                       std::vector<double> strikePriceList, std::vector<double> riskFreeRateList,
                       std::vector<double> dividendYieldList, std::vector<double> volatilityList,
                       std::vector<double> timeToMaturityList, std::vector<unsigned int> requiredNumSamples,
                       double dowDivisor, py::list outputResults) {
            int retval;
            unsigned int numAssets = stockPriceList.size(); // use the length of the stock price list to determine how
                                                            // many assets we are dealing with...
            double DJIAoptionOutput;

            py::scoped_ostream_redirect outStream(std::cout, py::module::import("sys").attr("stdout"));

            retval =
                self.run(optionTypeList.data(), stockPriceList.data(), strikePriceList.data(), riskFreeRateList.data(),
                         dividendYieldList.data(), volatilityList.data(), timeToMaturityList.data(),
                         requiredNumSamples.data(), numAssets, dowDivisor, &DJIAoptionOutput);

            outputResults.append(DJIAoptionOutput);

            return retval;
        });

    py::class_<BinomialTreeInputDataType<double> >(m, "BinomialTreeInputDataTypeDouble")
        .def(py::init())
        .def_readwrite("S", &BinomialTreeInputDataType<double>::S)
        .def_readwrite("K", &BinomialTreeInputDataType<double>::K)
        .def_readwrite("T", &BinomialTreeInputDataType<double>::T)
        .def_readwrite("rf", &BinomialTreeInputDataType<double>::rf)
        .def_readwrite("V", &BinomialTreeInputDataType<double>::V)
        .def_readwrite("q", &BinomialTreeInputDataType<double>::q)
        .def_readwrite("N", &BinomialTreeInputDataType<double>::N);

    py::class_<BinomialTree>(m, "BinomialTree")
        .def(py::init<std::string>())
        .def("claimDevice", &BinomialTree::claimDevice, py::call_guard<py::scoped_ostream_redirect>())
        .def("releaseDevice", &BinomialTree::releaseDevice, py::call_guard<py::scoped_ostream_redirect>())
        .def("deviceIsPrepared", &BinomialTree::deviceIsPrepared, py::call_guard<py::scoped_ostream_redirect>())
        .def("lastruntime", &BinomialTree::getLastRunTime, py::call_guard<py::scoped_ostream_redirect>())

        .def_readonly_static("MaxNodeDepth", &BinomialTreeMaxNodeDepth)
        .def_readonly_static("OptionTypeEuropeanPut", &BinomialTreeEuropeanPut)
        .def_readonly_static("OptionTypeEuropeanCall", &BinomialTreeEuropeanCall)
        .def_readonly_static("OptionTypeAmericanPut", &BinomialTreeAmericanPut)
        .def_readonly_static("OptionTypeAmericanCall", &BinomialTreeAmericanCall)

        .def("run", [](BinomialTree& self, std::vector<BinomialTreeInputDataType<double> > inputData,
                       py::list outputResults, int optionType) {
            int retval;
            unsigned int numAssets = inputData.size();
            std::vector<double> optionPriceVector(numAssets);

            py::scoped_ostream_redirect outStream(std::cout, py::module::import("sys").attr("stdout"));

            retval = self.run(inputData.data(), optionPriceVector.data(), optionType, numAssets);

            // add the output data to the specified py::list so it is available to
            // the calling function....
            for (unsigned int i = 0; i < optionPriceVector.size(); i++) {
                outputResults.append(optionPriceVector[i]);
            }

            return retval;
        });

    py::class_<FDHeston>(m, "FDHeston")

        .def(py::init<unsigned int, unsigned int, std::string>())

        .def("claimDevice", &FDHeston::claimDevice, py::call_guard<py::scoped_ostream_redirect>())
        .def("releaseDevice", &FDHeston::releaseDevice, py::call_guard<py::scoped_ostream_redirect>())
        .def("deviceIsPrepared", &FDHeston::deviceIsPrepared, py::call_guard<py::scoped_ostream_redirect>())
        .def("lastruntime", &FDHeston::getLastRunTime, py::call_guard<py::scoped_ostream_redirect>())

        .def(
            "run",
            [](FDHeston& self, double stockPrice_S, double strikePrice_K, double riskFreeDomesticInterestRate_rd,
               double volatility_V, double timeToMaturity, double meanReversionRate_kappa,
               double volatilityOfVolatility_sigma, double correlationCoefficient_rho, double longRunAveragePrice_eta) {
                int retval;
                // so in this first case following the logic and order of the .h file will be a single value output as
                // pOptionPrice - query why no numsteps - termination criteria //
                double optionPrice;
                py::scoped_ostream_redirect outStream(std::cout, py::module::import("sys").attr("stdout"));
                retval = self.run(stockPrice_S, strikePrice_K, riskFreeDomesticInterestRate_rd, volatility_V,
                                  timeToMaturity, meanReversionRate_kappa, volatilityOfVolatility_sigma,
                                  correlationCoefficient_rho, longRunAveragePrice_eta, &optionPrice);
                return std::make_tuple(retval, optionPrice);
            })

        .def("run",
             [](FDHeston& self, double stockPrice_S, double strikePrice_K, double riskFreeDomesticInterestRate_rd,
                double volatility_V, double timeToMaturity, double meanReversionRate_kappa,
                double volatilityOfVolatility_sigma, double correlationCoefficient_rho, double longRunAveragePrice_eta,
                int NumSteps) {
                 int retval;
                 // so in this second case following the logic and order of the .h file will be a single value output as
                 // pOptionPrice - with numsteps //
                 double optionPrice;
                 py::scoped_ostream_redirect outStream(std::cout, py::module::import("sys").attr("stdout"));
                 retval = self.run(stockPrice_S, strikePrice_K, riskFreeDomesticInterestRate_rd, volatility_V,
                                   timeToMaturity, meanReversionRate_kappa, volatilityOfVolatility_sigma,
                                   correlationCoefficient_rho, longRunAveragePrice_eta, NumSteps, &optionPrice);
                 return std::make_tuple(retval, optionPrice);
             })

        .def("run", [](FDHeston& self, double stockPrice_S, double strikePrice_K,
                       double riskFreeDomesticInterestRate_rd, double volatility_V, double timeToMaturity,
                       double meanReversionRate_kappa, double volatilityOfVolatility_sigma,
                       double correlationCoefficient_rho, double longRunAveragePrice_eta, int NumSteps, int sGridsize,
                       int vGridsize, py::list sGridOutput, py::list vGridOutput, py::list priceGridOutput) {
            int retval;
            // so in this third case following the logic and order of the .h file will there be an array of values
            // output ? //
            std::vector<double> sGridVector(sGridsize); // 128 understand this is fixed by the bitstream build, so
                                                        // bitstream would need to be rebuilt if the value is not 128 //
            std::vector<double> vGridVector(vGridsize); // 64 understand this is fixed by the bitstream build, so
                                                        // bitstream would need to be rebuilt if the value is not 64 //
            std::vector<double> priceGridVector(sGridsize * vGridsize); // 128 * 64  //

            py::scoped_ostream_redirect outStream(std::cout, py::module::import("sys").attr("stdout"));
            retval =
                self.run(stockPrice_S, strikePrice_K, riskFreeDomesticInterestRate_rd, volatility_V, timeToMaturity,
                         meanReversionRate_kappa, volatilityOfVolatility_sigma, correlationCoefficient_rho,
                         longRunAveragePrice_eta, NumSteps, priceGridVector, sGridVector, vGridVector);
            for (auto i : priceGridVector) priceGridOutput.append(i);
            for (auto i : sGridVector) sGridOutput.append(i);
            for (auto i : vGridVector) vGridOutput.append(i);
            return retval;
        });

    py::class_<PopMCMC>(m, "PopMCMC")

        .def(py::init<std::string>())

        .def("claimDevice", &PopMCMC::claimDevice, py::call_guard<py::scoped_ostream_redirect>())
        .def("releaseDevice", &PopMCMC::releaseDevice, py::call_guard<py::scoped_ostream_redirect>())
        .def("deviceIsPrepared", &PopMCMC::deviceIsPrepared, py::call_guard<py::scoped_ostream_redirect>())
        .def("lastruntime", &PopMCMC::getLastRunTime, py::call_guard<py::scoped_ostream_redirect>())

        .def("run", [](PopMCMC& self, int samples, int burninSamples, double sigma, py::list output) {
            int retval;

            // size of the results returned is the samples minus the burn in
            std::vector<double> outputVector(samples - burninSamples);
            py::scoped_ostream_redirect outStream(std::cout, py::module::import("sys").attr("stdout"));

            retval = self.run(samples, burninSamples, sigma, outputVector.data());
            for (auto i : outputVector) output.append(i);

            return retval;
        });

    py::class_<CFBlackScholes>(m, "CFBlackScholes")
        .def(py::init<unsigned int, std::string>())

        .def("claimDevice", &CFBlackScholes::claimDevice, py::call_guard<py::scoped_ostream_redirect>())
        .def("releaseDevice", &CFBlackScholes::releaseDevice, py::call_guard<py::scoped_ostream_redirect>())
        .def("deviceIsPrepared", &CFBlackScholes::deviceIsPrepared, py::call_guard<py::scoped_ostream_redirect>())
        .def("lastruntime", &CFBlackScholes::getLastRunTime)

        .def("run", [](CFBlackScholes& self, std::vector<float> stockPriceList, std::vector<float> strikePriceList,
                       std::vector<float> volatilityList, std::vector<float> riskFreeRateList,
                       std::vector<float> timeToMaturityList,
                       // Above are Input Buffers   - Below are Output Buffers
                       py::list optionPriceList, py::list deltaList, py::list gammaList, py::list vegaList,
                       py::list thetaList, py::list rhoList,
                       // Underneath is just the format chosen, as using the C++ example
                       OptionType optionType, unsigned int numAssets)

             {
                 int retval;

                 py::scoped_ostream_redirect outStream(std::cout, py::module::import("sys").attr("stdout"));
                 for (unsigned int i = 0; i < numAssets; i++) {
                     self.stockPrice[i] = stockPriceList[i];
                     self.strikePrice[i] = strikePriceList[i];
                     self.volatility[i] = volatilityList[i];
                     self.riskFreeRate[i] = riskFreeRateList[i];
                     self.timeToMaturity[i] = timeToMaturityList[i];
                 }
                 retval = self.run(optionType, numAssets);

                 // so after the execution these should be filled with results -> transfer to python lists
                 for (unsigned int i = 0; i < numAssets; i++) {
                     optionPriceList.append(self.optionPrice[i]);
                     deltaList.append(self.delta[i]);
                     gammaList.append(self.gamma[i]);
                     vegaList.append(self.vega[i]);
                     thetaList.append(self.theta[i]);
                     rhoList.append(self.rho[i]);
                 }

                 return retval;
             });

    py::class_<CFBlackScholesMerton>(m, "CFBlackScholesMerton")
        .def(py::init<unsigned int, std::string>())

        .def("claimDevice", &CFBlackScholesMerton::claimDevice, py::call_guard<py::scoped_ostream_redirect>())
        .def("releaseDevice", &CFBlackScholesMerton::releaseDevice, py::call_guard<py::scoped_ostream_redirect>())
        .def("deviceIsPrepared", &CFBlackScholesMerton::deviceIsPrepared, py::call_guard<py::scoped_ostream_redirect>())
        .def("lastruntime", &CFBlackScholesMerton::getLastRunTime)

        .def("run",
             [](CFBlackScholesMerton& self, std::vector<float> stockPriceList, std::vector<float> strikePriceList,
                std::vector<float> volatilityList, std::vector<float> riskFreeRateList,
                std::vector<float> timeToMaturityList, std::vector<float> dividendYieldList,
                // Above are Input Buffers   - Below are Output Buffers
                py::list optionPriceList, py::list deltaList, py::list gammaList, py::list vegaList, py::list thetaList,
                py::list rhoList,
                // Underneath is just the format chosen, as using the C++ example
                OptionType optionType, unsigned int numAssets)

             {
                 int retval;

                 py::scoped_ostream_redirect outStream(std::cout, py::module::import("sys").attr("stdout"));
                 for (unsigned int i = 0; i < numAssets; i++) {
                     self.stockPrice[i] = stockPriceList[i];
                     self.strikePrice[i] = strikePriceList[i];
                     self.volatility[i] = volatilityList[i];
                     self.riskFreeRate[i] = riskFreeRateList[i];
                     self.timeToMaturity[i] = timeToMaturityList[i];
                     self.dividendYield[i] = dividendYieldList[i];
                 }
                 retval = self.run(optionType, numAssets);

                 // so after the execution these should be filled with results -> transfer to python lists
                 for (unsigned int i = 0; i < numAssets; i++) {
                     optionPriceList.append(self.optionPrice[i]);
                     deltaList.append(self.delta[i]);
                     gammaList.append(self.gamma[i]);
                     vegaList.append(self.vega[i]);
                     thetaList.append(self.theta[i]);
                     rhoList.append(self.rho[i]);
                 }

                 return retval;
             });

    py::class_<CFQuanto>(m, "Quanto")
        .def(py::init<unsigned int, std::string>())

        .def("claimDevice", &CFQuanto::claimDevice, py::call_guard<py::scoped_ostream_redirect>())
        .def("releaseDevice", &CFQuanto::releaseDevice, py::call_guard<py::scoped_ostream_redirect>())
        .def("deviceIsPrepared", &CFQuanto::deviceIsPrepared, py::call_guard<py::scoped_ostream_redirect>())
        .def("lastruntime", &CFQuanto::getLastRunTime)

        .def("run", [](CFQuanto& self, std::vector<float> stockPriceList, std::vector<float> strikePriceList,
                       std::vector<float> volatilityList, std::vector<float> timeToMaturityList,
                       std::vector<float> domesticRateList, std::vector<float> foreignRateList,
                       std::vector<float> dividendYieldList, std::vector<float> exchangeRateList,
                       std::vector<float> exchangeRateVolatilityList, std::vector<float> correlationList,
                       // Above are Input Buffers   - Below are Output Buffers
                       py::list optionPriceList, py::list deltaList, py::list gammaList, py::list vegaList,
                       py::list thetaList, py::list rhoList,
                       // Underneath are the values passed in as part of the call
                       OptionType optionType, unsigned int numAssets)

             {
                 int retval;

                 py::scoped_ostream_redirect outStream(std::cout, py::module::import("sys").attr("stdout"));
                 for (unsigned int i = 0; i < numAssets; i++) {
                     self.stockPrice[i] = stockPriceList[i];
                     self.strikePrice[i] = strikePriceList[i];
                     self.volatility[i] = volatilityList[i];
                     self.timeToMaturity[i] = timeToMaturityList[i];
                     self.domesticRate[i] = domesticRateList[i];
                     self.foreignRate[i] = foreignRateList[i];
                     self.dividendYield[i] = dividendYieldList[i];
                     self.exchangeRate[i] = exchangeRateList[i];
                     self.exchangeRateVolatility[i] = exchangeRateVolatilityList[i];
                     self.correlation[i] = correlationList[i];
                 }
                 retval = self.run(optionType, numAssets);

                 // so after the execution these should be filled with results -> transfer to python lists
                 for (unsigned int i = 0; i < numAssets; i++) {
                     optionPriceList.append(self.optionPrice[i]);
                     deltaList.append(self.delta[i]);
                     gammaList.append(self.gamma[i]);
                     vegaList.append(self.vega[i]);
                     thetaList.append(self.theta[i]);
                     rhoList.append(self.rho[i]);
                 }

                 return retval;
             });

    py::class_<m76::m76_input_data>(m, "m76_input_data")
        .def(py::init())
        .def_readwrite("S", &m76::m76_input_data::S)
        .def_readwrite("sigma", &m76::m76_input_data::sigma)
        .def_readwrite("K", &m76::m76_input_data::K)
        .def_readwrite("r", &m76::m76_input_data::r)
        .def_readwrite("T", &m76::m76_input_data::T)
        .def_readwrite("lamb", &m76::m76_input_data::lambda)
        .def_readwrite("kappa", &m76::m76_input_data::kappa)
        .def_readwrite("delta", &m76::m76_input_data::delta);

    py::class_<m76>(m, "m76")
        .def(py::init<std::string>())

        .def("claimDevice", &m76::claimDevice, py::call_guard<py::scoped_ostream_redirect>())
        .def("releaseDevice", &m76::releaseDevice, py::call_guard<py::scoped_ostream_redirect>())
        .def("deviceIsPrepared", &m76::deviceIsPrepared, py::call_guard<py::scoped_ostream_redirect>())
        // missing     .def("lastruntime", &m76::getLastRunTime)

        .def("run", [](m76& self, std::vector<m76::m76_input_data> inputData,
                       // Above are Input Buffers - Below is Output Buffer
                       py::list outputData,
                       // Underneath are the values passed in as part of the call
                       int numOptions)

             {
                 int retval;
                 std::vector<float> optionPriceVector(numOptions);
                 // unsigned int numAssets = inputData.size();

                 py::scoped_ostream_redirect outStream(std::cout, py::module::import("sys").attr("stdout"));

                 retval = self.run(inputData.data(), optionPriceVector.data(), numOptions);

                 // so after the execution these should be filled with results -> transfer to python lists
                 for (int i = 0; i < numOptions; i++) {
                     // optionPriceList.append(self.outputData[i]);
                     outputData.append(optionPriceVector[i]);
                 }

                 return retval;
             });

    py::class_<hcf::hcf_input_data>(m, "hcf_input_data")
        .def(py::init())
        .def_readwrite("s0", &hcf::hcf_input_data::s0)
        .def_readwrite("v0", &hcf::hcf_input_data::v0)
        .def_readwrite("K", &hcf::hcf_input_data::K)
        .def_readwrite("rho", &hcf::hcf_input_data::rho)
        .def_readwrite("T", &hcf::hcf_input_data::T)
        .def_readwrite("r", &hcf::hcf_input_data::r)
        .def_readwrite("kappa", &hcf::hcf_input_data::kappa)
        .def_readwrite("vvol", &hcf::hcf_input_data::vvol)
        .def_readwrite("vbar", &hcf::hcf_input_data::vbar);

    py::class_<hcf>(m, "hcf")
        .def(py::init<std::string>())

        .def("claimDevice", &hcf::claimDevice, py::call_guard<py::scoped_ostream_redirect>())
        .def("releaseDevice", &hcf::releaseDevice, py::call_guard<py::scoped_ostream_redirect>())
        .def("deviceIsPrepared", &hcf::deviceIsPrepared, py::call_guard<py::scoped_ostream_redirect>())
        // missing     .def("lastruntime", &hcf::getLastRunTime)

        .def("run", [](hcf& self, std::vector<hcf::hcf_input_data> inputData,
                       // Above are Input Buffers - Below is Output Buffer
                       py::list outputData,
                       // Underneath are the values passed in as part of the call
                       int numOptions)

             {
                 int retval;
                 std::vector<float> optionPriceVector(numOptions);
                 // unsigned int numAssets = inputData.size();

                 py::scoped_ostream_redirect outStream(std::cout, py::module::import("sys").attr("stdout"));

                 retval = self.run(inputData.data(), optionPriceVector.data(), numOptions);

                 // so after the execution these should be filled with results -> transfer to python lists
                 for (int i = 0; i < numOptions; i++) {
                     // optionPriceList.append(self.outputData[i]);
                     outputData.append(optionPriceVector[i]);
                 }

                 return retval;
             });

    py::class_<HJM>(m, "HJM")
        .def(py::init<std::string>())

        .def("claimDevice", &HJM::claimDevice, py::call_guard<py::scoped_ostream_redirect>())
        .def("releaseDevice", &HJM::releaseDevice, py::call_guard<py::scoped_ostream_redirect>())
        .def("deviceIsPrepared", &HJM::deviceIsPrepared, py::call_guard<py::scoped_ostream_redirect>())
        .def("lastruntime", &HJM::getLastRunTime)

        .def("run", [](HJM& self, std::vector<double> historicalData, std::vector<unsigned> seeds,
                       // Above are Input Buffers - Below is Output Buffer
                       py::list outputPrice,
                       // Underneath are the values passed in as part of the call
                       unsigned noTenors, unsigned noCurves, unsigned noPaths, float simYears, float zcbMaturity) {
            double outPrice;

            py::scoped_ostream_redirect outStream(std::cout, py::module::import("sys").attr("stdout"));

            int retval = self.run(historicalData.data(), noTenors, noCurves, noPaths, simYears, zcbMaturity,
                                  seeds.data(), &outPrice);
            outputPrice.append(outPrice);
            return retval;
        });

    py::class_<LMM>(m, "LMM")
        .def(py::init<std::string>())

        .def("claimDeviceCap", &LMM::claimDeviceCap, py::call_guard<py::scoped_ostream_redirect>())
        .def("claimDeviceRatchetFloater", &LMM::claimDeviceRatchetFloater,
             py::call_guard<py::scoped_ostream_redirect>())
        .def("claimDeviceRatchetCap", &LMM::claimDeviceRatchetCap, py::call_guard<py::scoped_ostream_redirect>())
        .def("releaseDevice", &LMM::releaseDevice, py::call_guard<py::scoped_ostream_redirect>())
        .def("lastruntime", &LMM::getLastRunTime)

        .def("runCap",
             [](LMM& self, std::vector<float> presentRate, std::vector<float> capletVolas, std::vector<unsigned> seeds,
                // Above are Input Buffers - Below is Output Buffer
                py::list outputPrice,
                // Underneath are the values passed in as part of the call
                unsigned noTenors, unsigned noPaths, float rhoBeta, float notional, float caprate) {
                 float outPrice;

                 py::scoped_ostream_redirect outStream(std::cout, py::module::import("sys").attr("stdout"));

                 int retval = self.runCap(noTenors, noPaths, presentRate.data(), rhoBeta, capletVolas.data(), notional,
                                          caprate, seeds.data(), &outPrice);
                 outputPrice.append(outPrice);
                 return retval;
             })

        .def("runRatchetCap",
             [](LMM& self, std::vector<float> presentRate, std::vector<float> capletVolas, std::vector<unsigned> seeds,
                // Above are Input Buffers - Below is Output Buffer
                py::list outputPrice,
                // Underneath are the values passed in as part of the call
                unsigned noTenors, unsigned noPaths, float rhoBeta, float notional, float spread, float kappa0) {
                 float outPrice;

                 py::scoped_ostream_redirect outStream(std::cout, py::module::import("sys").attr("stdout"));

                 int retval = self.runRatchetCap(noTenors, noPaths, presentRate.data(), rhoBeta, capletVolas.data(),
                                                 notional, spread, kappa0, seeds.data(), &outPrice);
                 outputPrice.append(outPrice);
                 return retval;
             })

        .def("runRatchetFloater",
             [](LMM& self, std::vector<float> presentRate, std::vector<float> capletVolas, std::vector<unsigned> seeds,
                // Above are Input Buffers - Below is Output Buffer
                py::list outputPrice,
                // Underneath are the values passed in as part of the call
                unsigned noTenors, unsigned noPaths, float rhoBeta, float notional, float rfX, float rfY,
                float rfAlpha) {
                 float outPrice;

                 py::scoped_ostream_redirect outStream(std::cout, py::module::import("sys").attr("stdout"));

                 int retval = self.runRatchetFloater(noTenors, noPaths, presentRate.data(), rhoBeta, capletVolas.data(),
                                                     notional, rfX, rfY, rfAlpha, seeds.data(), &outPrice);
                 outputPrice.append(outPrice);
                 return retval;
             });

    py::class_<CFGarmanKohlhagen>(m, "CFGarmanKohlhagen")
        .def(py::init<unsigned int, std::string>())

        .def("claimDevice", &CFGarmanKohlhagen::claimDevice, py::call_guard<py::scoped_ostream_redirect>())
        .def("releaseDevice", &CFGarmanKohlhagen::releaseDevice, py::call_guard<py::scoped_ostream_redirect>())
        .def("deviceIsPrepared", &CFGarmanKohlhagen::deviceIsPrepared, py::call_guard<py::scoped_ostream_redirect>())
        .def("lastruntime", &CFGarmanKohlhagen::getLastRunTime)

        .def("run", [](CFGarmanKohlhagen& self, std::vector<float> stockPriceList, std::vector<float> strikePriceList,
                       std::vector<float> volatilityList, std::vector<float> timeToMaturityList,
                       std::vector<float> domesticRateList, std::vector<float> foreignRateList,
                       // std::vector<float> dividendYieldList, std::vector<float> exchangeRateList,
                       // std::vector<float> exchangeRateVolatilityList, std::vector<float> correlationList,
                       // Above are Input Buffers   - Below are Output Buffers
                       py::list optionPriceList, py::list deltaList, py::list gammaList, py::list vegaList,
                       py::list thetaList, py::list rhoList,
                       // Underneath are the values passed in as part of the call
                       OptionType optionType, unsigned int numAssets)

             {
                 int retval;

                 py::scoped_ostream_redirect outStream(std::cout, py::module::import("sys").attr("stdout"));
                 for (unsigned int i = 0; i < numAssets; i++) {
                     self.stockPrice[i] = stockPriceList[i];
                     self.strikePrice[i] = strikePriceList[i];
                     self.volatility[i] = volatilityList[i];
                     self.timeToMaturity[i] = timeToMaturityList[i];
                     self.domesticRate[i] = domesticRateList[i];
                     self.foreignRate[i] = foreignRateList[i];
                     // self.dividendYield[i] = dividendYieldList[i];
                     // self.exchangeRate[i] = exchangeRateList[i];
                     // self.exchangeRateVolatility[i] = exchangeRateVolatilityList[i];
                     // self.correlation[i] = correlationList[i];
                 }
                 retval = self.run(optionType, numAssets);

                 // so after the execution these should be filled with results -> transfer to python lists
                 for (unsigned int i = 0; i < numAssets; i++) {
                     optionPriceList.append(self.optionPrice[i]);
                     deltaList.append(self.delta[i]);
                     gammaList.append(self.gamma[i]);
                     vegaList.append(self.vega[i]);
                     thetaList.append(self.theta[i]);
                     rhoList.append(self.rho[i]);
                 }

                 return retval;
             });

    py::class_<fdbslv>(m, "fdbslv")
        .def(py::init<unsigned int, unsigned int, std::string>())
        .def("claimDevice", &fdbslv::claimDevice, py::call_guard<py::scoped_ostream_redirect>())
        .def("releaseDevice", &fdbslv::releaseDevice, py::call_guard<py::scoped_ostream_redirect>())
        .def("deviceIsPrepared", &fdbslv::deviceIsPrepared, py::call_guard<py::scoped_ostream_redirect>())
        // .def("lastruntime", &fdbslv::getLastRunTime)

        .def("run",
             [](fdbslv& self, std::vector<float> xGridList, std::vector<float> tGridList, std::vector<float> sigmaList,
                std::vector<float> rateList, std::vector<float> initial_conditionsList, double solverTheta,
                double boundaryLower, double boundaryUpper,
                // Above are Input Buffers - Below is the Output Buffer
                py::list outputResultList)

             {
                 int retval;
                 unsigned int solverNderived =
                     xGridList.size(); // use the length of the xGrid list to derive the solverN value
                 std::vector<float> solution(solverNderived);

                 retval = self.run(xGridList, tGridList, sigmaList, rateList, initial_conditionsList, solverTheta,
                                   boundaryLower, boundaryUpper, solution);

                 outputResultList.append(solution);
                 return retval;
             });

    py::class_<CFB76>(m, "CFB76")
        .def(py::init<unsigned int, std::string>())
        // .def(py::init<std::string,unsigned int>())
        // .def(py::init<unsigned int>())

        .def("claimDevice", &CFB76::claimDevice, py::call_guard<py::scoped_ostream_redirect>())
        .def("releaseDevice", &CFB76::releaseDevice, py::call_guard<py::scoped_ostream_redirect>())
        .def("deviceIsPrepared", &CFB76::deviceIsPrepared, py::call_guard<py::scoped_ostream_redirect>())
        .def("lastruntime", &CFB76::getLastRunTime)

        .def("run", [](CFB76& self, std::vector<float> forwardPriceList, std::vector<float> strikePriceList,
                       std::vector<float> volatilityList, std::vector<float> riskFreeRateList,
                       std::vector<float> timeToMaturityList,
                       // Above are Input Buffers   - Below are Output Buffers
                       py::list optionPriceList, py::list deltaList, py::list gammaList, py::list vegaList,
                       py::list thetaList, py::list rhoList,
                       // Underneath is just the format chosen, as using the C++ example
                       OptionType optionType, unsigned int numAssets)

             {
                 int retval;

                 py::scoped_ostream_redirect outStream(std::cout, py::module::import("sys").attr("stdout"));
                 for (unsigned int i = 0; i < numAssets; i++) {
                     self.forwardPrice[i] = forwardPriceList[i];
                     self.strikePrice[i] = strikePriceList[i];
                     self.volatility[i] = volatilityList[i];
                     self.riskFreeRate[i] = riskFreeRateList[i];
                     self.timeToMaturity[i] = timeToMaturityList[i];
                 }
                 retval = self.run(optionType, numAssets);

                 // so after the execution these should be filled with results -> transfer to python lists
                 for (unsigned int i = 0; i < numAssets; i++) {
                     optionPriceList.append(self.optionPrice[i]);
                     deltaList.append(self.delta[i]);
                     gammaList.append(self.gamma[i]);
                     vegaList.append(self.vega[i]);
                     thetaList.append(self.theta[i]);
                     rhoList.append(self.rho[i]);
                 }

                 return retval;
             });

    py::class_<CreditDefaultSwap>(m, "CreditDefaultSwap")
        .def(py::init<std::string>())

        .def("claimDevice", &CreditDefaultSwap::claimDevice, py::call_guard<py::scoped_ostream_redirect>())
        .def("releaseDevice", &CreditDefaultSwap::releaseDevice, py::call_guard<py::scoped_ostream_redirect>())
        .def("deviceIsPrepared", &CreditDefaultSwap::deviceIsPrepared, py::call_guard<py::scoped_ostream_redirect>())
        .def("lastruntime", &CreditDefaultSwap::getLastRunTime, py::call_guard<py::scoped_ostream_redirect>())

        .def("run", [](CreditDefaultSwap& self, std::vector<float> timesIR, std::vector<float> ratesIR,
                       std::vector<float> timesHazard, std::vector<float> ratesHazard, std::vector<float> notional,
                       std::vector<float> recovery, std::vector<float> maturity, std::vector<int> frequency,
                       // Above are Input Buffers - Below is Output Buffer
                       py::list cdsSpread) {
            int numOutputs = recovery.size(); // use the length of the recovery list to derive the cdsSpread array size

            std::vector<float> cdsSpreadVector(numOutputs);

            py::scoped_ostream_redirect outStream(std::cout, py::module::import("sys").attr("stdout"));

            int retval =
                self.run(timesIR.data(), ratesIR.data(), timesHazard.data(), ratesHazard.data(), notional.data(),
                         recovery.data(), maturity.data(), frequency.data(), cdsSpreadVector.data());

            for (int i = 0; i < numOutputs; i++) {
                cdsSpread.append(cdsSpreadVector[i]);
            }
            return retval;
        });

    py::class_<portfolio_optimisation>(m, "portfolio_optimisation")
        .def(py::init<std::string>())
        .def("claimDevice", &portfolio_optimisation::claimDevice, py::call_guard<py::scoped_ostream_redirect>())
        .def("releaseDevice", &portfolio_optimisation::releaseDevice, py::call_guard<py::scoped_ostream_redirect>())
        .def("deviceIsPrepared", &portfolio_optimisation::deviceIsPrepared,
             py::call_guard<py::scoped_ostream_redirect>())
        //.def("lastruntime", &portfolio_optimisation::getLastRunTime) - missing

        .def("run", [](portfolio_optimisation& self, std::vector<float> testDataList, int num_prices, int num_assets,
                       float riskFreeRate, float targetReturn,
                       // Above are Inputs - Below are Outputs
                       py::list GMVPWeightsList, py::list GMVPVariance, py::list GMVPReturn, py::list EffWeightsList,
                       py::list EffVariance, py::list EffReturn, py::list TanWeightsList, py::list TanVariance,
                       py::list TanReturn, py::list TanSharpe, py::list EffTanWeightsList, py::list EffTanVariance,
                       py::list EffTanReturn) {

            float GMVPVarianceFloat;
            float GMVPReturnFloat;

            float EffVarianceFloat;
            float EffReturnFloat;

            float TanVarianceFloat;
            float TanReturnFloat;
            float TanSharpeFloat;

            float EffTanVarianceFloat;
            float EffTanReturnFloat;

            std::vector<float> GMVPWeightsVector(num_assets);
            std::vector<float> EffWeightsVector(num_assets);
            std::vector<float> TanWeightsVector(num_assets);
            std::vector<float> EffTanWeightsVector(num_assets);

            py::scoped_ostream_redirect outStream(std::cout, py::module::import("sys").attr("stdout"));

            int retval =
                self.run(testDataList.data(), num_prices, num_assets, riskFreeRate, targetReturn, GMVPWeightsVector,
                         &GMVPVarianceFloat, &GMVPReturnFloat, EffWeightsVector, &EffVarianceFloat, &EffReturnFloat,
                         TanWeightsVector, &TanVarianceFloat, &TanReturnFloat, &TanSharpeFloat, EffTanWeightsVector,
                         &EffTanVarianceFloat, &EffTanReturnFloat);

            // Return results in pylist
            GMVPVariance.append(GMVPVarianceFloat);
            GMVPReturn.append(GMVPReturnFloat);
            EffVariance.append(EffVarianceFloat);
            EffReturn.append(EffReturnFloat);
            TanVariance.append(TanVarianceFloat);
            TanReturn.append(TanSharpeFloat);
            TanSharpe.append(TanSharpeFloat);
            EffTanVariance.append(EffTanVarianceFloat);
            EffTanReturn.append(EffTanReturnFloat);

            for (int i = 0; i < num_assets; i++) {
                GMVPWeightsList.append(GMVPWeightsVector[i]);
                EffWeightsList.append(EffWeightsVector[i]);
                TanWeightsList.append(TanWeightsVector[i]);
                EffTanWeightsList.append(EffTanWeightsVector[i]);
            }
            return retval;
        });

    py::class_<HullWhiteAnalytic>(m, "HullWhiteAnalytic")
        .def(py::init<std::string>())

        .def("claimDevice", &HullWhiteAnalytic::claimDevice, py::call_guard<py::scoped_ostream_redirect>())
        .def("releaseDevice", &HullWhiteAnalytic::releaseDevice, py::call_guard<py::scoped_ostream_redirect>())
        .def("deviceIsPrepared", &HullWhiteAnalytic::deviceIsPrepared, py::call_guard<py::scoped_ostream_redirect>())
        .def("lastruntime", &HullWhiteAnalytic::getLastRunTime, py::call_guard<py::scoped_ostream_redirect>())

        .def("run", [](HullWhiteAnalytic& self, double a, double sigma, std::vector<double> times,
                       std::vector<double> rates, std::vector<double> t, std::vector<double> T,
                       // Above are Input Buffers - Below is Output Buffer
                       py::list P) {
            int numOutputs = t.size(); // use the length of the recovery list to derive the output array size

            std::vector<double> PVector(numOutputs);

            py::scoped_ostream_redirect outStream(std::cout, py::module::import("sys").attr("stdout"));

            int retval = self.run(a, sigma, times.data(), rates.data(), t.data(), T.data(), PVector.data());

            for (int i = 0; i < numOutputs; i++) {
                P.append(PVector[i]);
            }
            return retval;
        });

#ifdef VERSION_INFO
    m.attr("__version__") = VERSION_INFO;
#else
    m.attr("__version__") = "dev";
#endif
}
