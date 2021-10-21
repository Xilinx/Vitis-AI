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

#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>

#include <ql/qldefines.hpp>

#ifdef BOOST_MSVC
#include <ql/auto_link.hpp>
#endif
#include <ql/time/calendars/target.hpp>
#include <ql/time/daycounters/actual360.hpp>
#include <ql/time/daycounters/actualactual.hpp>

// Cox-Ross-Rubinstein
#include <ql/instruments/europeanoption.hpp>
#include <ql/pricingengines/vanilla/analyticeuropeanengine.hpp>
#include <ql/pricingengines/vanilla/binomialengine.hpp>
#include <ql/termstructures/volatility/equityfx/blackconstantvol.hpp>

// include the engine for struct definition
#include "xf_fintech/bt_engine.hpp"
#include "bt_testcases.hpp"

using namespace std;
using namespace xf::fintech;
using namespace QuantLib;
using namespace xf::fintech::internal;

// Quanlib built with doubles
typedef double DT;

#define MAX_OPTION_CALCULATIONS (128)

static const std::string inputTestCasesFileName = "../" + TestCasesFileName;
static const std::string inputSVGridFileName = "../" + SVGridFileName;

std::map<int, std::string> optionTypeFilePath = {
    {BinomialTreeEuropeanPut, "../" + BinomialTreeEuropeanPutName + "/"},
    {BinomialTreeEuropeanCall, "../" + BinomialTreeEuropeanCallName + "/"},
    {BinomialTreeAmericanPut, "../" + BinomialTreeAmericanPutName + "/"},
    {BinomialTreeAmericanCall, "../" + BinomialTreeAmericanCallName + "/"}};

static DT QuantlibEuropeanPut(BinomialTreeInputDataType<DT>* inputData) {
    Option::Type optionType = Option::Put;
    Calendar calendar = TARGET();
    DayCounter dayCounter = Actual360();
    Date today = Date::todaysDate();
    Settings::instance().evaluationDate() = today;
    Date maturity = today + Integer(inputData->T * 360 + 0.5);
    DT NPV = 0.0;

    Handle<Quote> underlyingH(boost::shared_ptr<Quote>(new SimpleQuote(inputData->S)));
    Handle<YieldTermStructure> flatTermStructure(
        boost::shared_ptr<YieldTermStructure>(new FlatForward(today, inputData->rf, dayCounter)));
    Handle<YieldTermStructure> flatDividendTS(
        boost::shared_ptr<YieldTermStructure>(new FlatForward(today, inputData->q, dayCounter)));
    Handle<BlackVolTermStructure> flatVolTS(
        boost::shared_ptr<BlackVolTermStructure>(new BlackConstantVol(today, calendar, inputData->V, dayCounter)));
    boost::shared_ptr<StrikedTypePayoff> payoff(new PlainVanillaPayoff(optionType, inputData->K));
    boost::shared_ptr<BlackScholesMertonProcess> bsmProcess(
        new BlackScholesMertonProcess(underlyingH, flatDividendTS, flatTermStructure, flatVolTS));

    // run engine
    ext::shared_ptr<PricingEngine> engine(new BinomialVanillaEngine<CoxRossRubinstein>(bsmProcess, inputData->N));
    boost::shared_ptr<Exercise> europeanExercise(new EuropeanExercise(maturity));
    VanillaOption option(payoff, europeanExercise);
    option.setPricingEngine(engine);
    NPV = option.NPV();

    return NPV;
}

static DT QuantlibEuropeanCall(BinomialTreeInputDataType<DT>* inputData) {
    Option::Type optionType = Option::Call;
    Calendar calendar = TARGET();
    DayCounter dayCounter = Actual360();
    Date today = Date::todaysDate();
    Settings::instance().evaluationDate() = today;
    Date maturity = today + Integer(inputData->T * 360 + 0.5);
    DT NPV = 0.0;

    Handle<Quote> underlyingH(boost::shared_ptr<Quote>(new SimpleQuote(inputData->S)));
    Handle<YieldTermStructure> flatTermStructure(
        boost::shared_ptr<YieldTermStructure>(new FlatForward(today, inputData->rf, dayCounter)));
    Handle<YieldTermStructure> flatDividendTS(
        boost::shared_ptr<YieldTermStructure>(new FlatForward(today, inputData->q, dayCounter)));
    Handle<BlackVolTermStructure> flatVolTS(
        boost::shared_ptr<BlackVolTermStructure>(new BlackConstantVol(today, calendar, inputData->V, dayCounter)));
    boost::shared_ptr<StrikedTypePayoff> payoff(new PlainVanillaPayoff(optionType, inputData->K));
    boost::shared_ptr<BlackScholesMertonProcess> bsmProcess(
        new BlackScholesMertonProcess(underlyingH, flatDividendTS, flatTermStructure, flatVolTS));

    // run engine
    ext::shared_ptr<PricingEngine> engine(new BinomialVanillaEngine<CoxRossRubinstein>(bsmProcess, inputData->N));
    boost::shared_ptr<Exercise> europeanExercise(new EuropeanExercise(maturity));
    VanillaOption option(payoff, europeanExercise);
    option.setPricingEngine(engine);
    NPV = option.NPV();

    return NPV;
}

static DT QuantlibAmericanPut(BinomialTreeInputDataType<DT>* inputData) {
    Option::Type optionType = Option::Put;
    Calendar calendar = TARGET();
    DayCounter dayCounter = Actual360();
    Date today = Date::todaysDate();
    Settings::instance().evaluationDate() = today;
    Date maturity = today + Integer(inputData->T * 360 + 0.5);
    DT NPV = 0.0;

    Handle<Quote> underlyingH(boost::shared_ptr<Quote>(new SimpleQuote(inputData->S)));
    Handle<YieldTermStructure> flatTermStructure(
        boost::shared_ptr<YieldTermStructure>(new FlatForward(today, inputData->rf, dayCounter)));
    Handle<YieldTermStructure> flatDividendTS(
        boost::shared_ptr<YieldTermStructure>(new FlatForward(today, inputData->q, dayCounter)));
    Handle<BlackVolTermStructure> flatVolTS(
        boost::shared_ptr<BlackVolTermStructure>(new BlackConstantVol(today, calendar, inputData->V, dayCounter)));
    boost::shared_ptr<StrikedTypePayoff> payoff(new PlainVanillaPayoff(optionType, inputData->K));
    boost::shared_ptr<BlackScholesMertonProcess> bsmProcess(
        new BlackScholesMertonProcess(underlyingH, flatDividendTS, flatTermStructure, flatVolTS));

    // run engine
    // ext::shared_ptr<PricingEngine> engine(new
    // BinomialVanillaEngine<CoxRossRubinstein>(bsmProcess, inputData->N));
    // using LeisenReimer for american options
    ext::shared_ptr<PricingEngine> engine(new BinomialVanillaEngine<LeisenReimer>(bsmProcess, inputData->N));
    boost::shared_ptr<Exercise> americanExercise(new AmericanExercise(today, maturity));
    VanillaOption option(payoff, americanExercise);
    option.setPricingEngine(engine);
    NPV = option.NPV();

    return NPV;
}

static DT QuantlibAmericanCall(BinomialTreeInputDataType<DT>* inputData) {
    Option::Type optionType = Option::Call;
    Calendar calendar = TARGET();
    DayCounter dayCounter = Actual360();
    Date today = Date::todaysDate();
    Settings::instance().evaluationDate() = today;
    Date maturity = today + Integer(inputData->T * 360 + 0.5);
    DT NPV = 0.0;

    Handle<Quote> underlyingH(boost::shared_ptr<Quote>(new SimpleQuote(inputData->S)));
    Handle<YieldTermStructure> flatTermStructure(
        boost::shared_ptr<YieldTermStructure>(new FlatForward(today, inputData->rf, dayCounter)));
    Handle<YieldTermStructure> flatDividendTS(
        boost::shared_ptr<YieldTermStructure>(new FlatForward(today, inputData->q, dayCounter)));
    Handle<BlackVolTermStructure> flatVolTS(
        boost::shared_ptr<BlackVolTermStructure>(new BlackConstantVol(today, calendar, inputData->V, dayCounter)));
    boost::shared_ptr<StrikedTypePayoff> payoff(new PlainVanillaPayoff(optionType, inputData->K));
    boost::shared_ptr<BlackScholesMertonProcess> bsmProcess(
        new BlackScholesMertonProcess(underlyingH, flatDividendTS, flatTermStructure, flatVolTS));

    // run engine
    // ext::shared_ptr<PricingEngine> engine(new
    // BinomialVanillaEngine<CoxRossRubinstein>(bsmProcess, inputData->N));
    // using LeisenReimer for american options
    ext::shared_ptr<PricingEngine> engine(new BinomialVanillaEngine<LeisenReimer>(bsmProcess, inputData->N));
    boost::shared_ptr<Exercise> americanExercise(new AmericanExercise(today, maturity));
    VanillaOption option(payoff, americanExercise);
    option.setPricingEngine(engine);
    NPV = option.NPV();

    return NPV;
}

static DT QuantlibBlackScholesCall(BinomialTreeInputDataType<DT>* inputData) {
    Option::Type optionType = Option::Call;
    Calendar calendar = TARGET();
    DayCounter dayCounter = Actual360();
    Date today = Date::todaysDate();
    Settings::instance().evaluationDate() = today;
    Date maturity = today + Integer(inputData->T * 360 + 0.5);
    DT NPV = 0.0;

    Handle<Quote> underlyingH(boost::shared_ptr<Quote>(new SimpleQuote(inputData->S)));
    Handle<YieldTermStructure> flatTermStructure(
        boost::shared_ptr<YieldTermStructure>(new FlatForward(today, inputData->rf, dayCounter)));
    Handle<YieldTermStructure> flatDividendTS(
        boost::shared_ptr<YieldTermStructure>(new FlatForward(today, inputData->q, dayCounter)));
    Handle<BlackVolTermStructure> flatVolTS(
        boost::shared_ptr<BlackVolTermStructure>(new BlackConstantVol(today, calendar, inputData->V, dayCounter)));
    boost::shared_ptr<StrikedTypePayoff> payoff(new PlainVanillaPayoff(optionType, inputData->K));
    boost::shared_ptr<BlackScholesMertonProcess> bsmProcess(
        new BlackScholesMertonProcess(underlyingH, flatDividendTS, flatTermStructure, flatVolTS));

    boost::shared_ptr<Exercise> europeanExercise(new EuropeanExercise(maturity));
    VanillaOption europeanOption(payoff, europeanExercise);

    // Black-Scholes for European
    std::string method;
    method = "Black-Scholes";
    europeanOption.setPricingEngine(ext::shared_ptr<PricingEngine>(new AnalyticEuropeanEngine(bsmProcess)));
    NPV = europeanOption.NPV();
    return NPV;
}

static DT QuantlibBlackScholesPut(BinomialTreeInputDataType<DT>* inputData) {
    Option::Type optionType = Option::Put;
    Calendar calendar = TARGET();
    DayCounter dayCounter = Actual360();
    Date today = Date::todaysDate();
    Settings::instance().evaluationDate() = today;
    Date maturity = today + Integer(inputData->T * 360 + 0.5);
    DT NPV = 0.0;

    Handle<Quote> underlyingH(boost::shared_ptr<Quote>(new SimpleQuote(inputData->S)));
    Handle<YieldTermStructure> flatTermStructure(
        boost::shared_ptr<YieldTermStructure>(new FlatForward(today, inputData->rf, dayCounter)));
    Handle<YieldTermStructure> flatDividendTS(
        boost::shared_ptr<YieldTermStructure>(new FlatForward(today, inputData->q, dayCounter)));
    Handle<BlackVolTermStructure> flatVolTS(
        boost::shared_ptr<BlackVolTermStructure>(new BlackConstantVol(today, calendar, inputData->V, dayCounter)));
    boost::shared_ptr<StrikedTypePayoff> payoff(new PlainVanillaPayoff(optionType, inputData->K));
    boost::shared_ptr<BlackScholesMertonProcess> bsmProcess(
        new BlackScholesMertonProcess(underlyingH, flatDividendTS, flatTermStructure, flatVolTS));

    boost::shared_ptr<Exercise> europeanExercise(new EuropeanExercise(maturity));
    VanillaOption europeanOption(payoff, europeanExercise);

    // Black-Scholes for European
    std::string method;
    method = "Black-Scholes";
    europeanOption.setPricingEngine(ext::shared_ptr<PricingEngine>(new AnalyticEuropeanEngine(bsmProcess)));
    NPV = europeanOption.NPV();
    return NPV;
}

void QuantlibBinomial(BinomialTreeInputDataType<DT>* inputData, DT* outputResult, int optionType, int numTests) {
    BinomialTreeInputDataType<DT> tempInputData[MAX_OPTION_CALCULATIONS];
    DT tempOutputResults[MAX_OPTION_CALCULATIONS];

    // Copy all test vectors from global memory to local memory
    for (int i = 0; i < numTests; i++) {
        tempInputData[i] = inputData[i];
    }

    // Calculate NPVs
    for (int i = 0; i < numTests; i++) {
        switch (optionType) {
            case BinomialTreeEuropeanPut:
                // using Quabtlib Black Scholes for European options
                tempOutputResults[i] = QuantlibBlackScholesPut(&tempInputData[i]);
                break;
            // tempOutputResults[i] = QuantlibEuropeanPut(&tempInputData[i]); break;

            case BinomialTreeEuropeanCall:
                // using Quabtlib Black Scholes for European options

                tempOutputResults[i] = QuantlibBlackScholesCall(&tempInputData[i]);
                break;
            // tempOutputResults[i] = QuantlibEuropeanCall(&tempInputData[i]);
            // break;

            case BinomialTreeAmericanPut:
                tempOutputResults[i] = QuantlibAmericanPut(&tempInputData[i]);
                break;
            case BinomialTreeAmericanCall:
                tempOutputResults[i] = QuantlibAmericanCall(&tempInputData[i]);
                break;

            default:
                break;
        }
    }

    // Copy results back to host
    for (int i = 0; i < numTests; i++) {
        outputResult[i] = tempOutputResults[i];
    }
}

int main(int argc, char* argv[]) {
    std::vector<BinomialTestCase<DT> > testcaseData;
    BinomialTestSVGrid<DT> testSVGrid;

    // testcase and sv grid input files
    ifstream inputFileTestCases;
    inputFileTestCases.open(inputTestCasesFileName);
    ifstream inputFileSVGrid;
    inputFileSVGrid.open(inputSVGridFileName);

    // set precision to 14 decimal places
    std::cout << std::fixed;
    std::cout << std::setprecision(14);

    try {
        BinomialTestCase<DT> tempTestCaseItem;
        std::string tmpStr;

        // skip first line as the comment
        std::getline(inputFileTestCases, tmpStr, '\n');

        while (!inputFileTestCases.eof()) {
            std::getline(inputFileTestCases, tmpStr, ',');
            if (tmpStr[0] == '#') break;
            tempTestCaseItem.name = tmpStr;
            std::getline(inputFileTestCases, tmpStr, ',');
            tempTestCaseItem.K = std::stod(tmpStr);
            std::getline(inputFileTestCases, tmpStr, ',');
            tempTestCaseItem.rf = std::stod(tmpStr);
            std::getline(inputFileTestCases, tmpStr, ',');
            tempTestCaseItem.T = std::stod(tmpStr);
            std::getline(inputFileTestCases, tmpStr, '\n');
            tempTestCaseItem.N = std::stoi(tmpStr);
            testcaseData.push_back(tempTestCaseItem);
        }

        // debug: output the testcases parsed
        int testCaseIndex = 0;
        for (std::vector<BinomialTestCase<DT> >::iterator it = testcaseData.begin(); it != testcaseData.end(); ++it) {
            std::cout << testCaseIndex << ": " << it->name << " " << it->K << " " << it->rf << " " << it->T << " "
                      << it->N << " " << std::endl;
            testCaseIndex++;
        }

        // skip first line as the comment
        std::getline(inputFileSVGrid, tmpStr, '\n');

        // next line contains the S grid values
        for (int i = 0; i < BINOMIAL_TESTCASE_NUM_S_GRID_VALUES - 1; i++) {
            std::getline(inputFileSVGrid, tmpStr, ',');
            testSVGrid.s[i] = std::stod(tmpStr);
        }
        std::getline(inputFileSVGrid, tmpStr, '\n');
        testSVGrid.s[BINOMIAL_TESTCASE_NUM_S_GRID_VALUES - 1] = std::stod(tmpStr);

        // next line contains the V grid values
        for (int i = 0; i < BINOMIAL_TESTCASE_NUM_V_GRID_VALUES - 1; i++) {
            std::getline(inputFileSVGrid, tmpStr, ',');
            testSVGrid.v[i] = std::stod(tmpStr);
        }
        std::getline(inputFileSVGrid, tmpStr, '\n');
        testSVGrid.v[BINOMIAL_TESTCASE_NUM_V_GRID_VALUES - 1] = std::stod(tmpStr);

        // debug: output the S & V grid parsed
        std::cout << "s grid: ";
        for (int i = 0; i < BINOMIAL_TESTCASE_NUM_S_GRID_VALUES; i++) {
            std::cout << testSVGrid.s[i] << " ";
        }
        std::cout << std::endl;

        std::cout << "v grid: ";
        for (int i = 0; i < BINOMIAL_TESTCASE_NUM_V_GRID_VALUES; i++) {
            std::cout << testSVGrid.v[i] << " ";
        }
        std::cout << std::endl;

        // repeat for each option type
        for (int k = BinomialTreeEuropeanPut; k < BinomialTreeAmericanCall + 1; k++) {
            // iterate around the test cases
            for (std::vector<BinomialTestCase<DT> >::iterator it = testcaseData.begin(); it != testcaseData.end();
                 ++it) {
                // populate structure based on the input data
                std::vector<BinomialTreeInputDataType<DT> > inputDataVector;
                ofstream outputFileTestCase;
                std::string outputFilename = optionTypeFilePath[k] + it->name + ".txt";
                std::cout << outputFilename << std::endl;
                outputFileTestCase.open(outputFilename);

                // loop around the s & v grid and populate
                for (int i = 0; i < BINOMIAL_TESTCASE_NUM_S_GRID_VALUES; i++) {
                    for (int j = 0; j < BINOMIAL_TESTCASE_NUM_V_GRID_VALUES; j++) {
                        BinomialTreeInputDataType<DT> tempInputData;
                        tempInputData.S = testSVGrid.s[i];
                        tempInputData.K = it->K;
                        tempInputData.T = it->T;
                        tempInputData.rf = it->rf;
                        tempInputData.V = testSVGrid.v[j];
                        tempInputData.q = 0;
                        tempInputData.N = it->N - 1;
                        inputDataVector.push_back(tempInputData);
                    }
                }

                BinomialTreeInputDataType<DT>* inputData = &inputDataVector[0];
                DT outputResult[MAX_OPTION_CALCULATIONS];
                int numberTests = BINOMIAL_TESTCASE_NUM_S_GRID_VALUES * BINOMIAL_TESTCASE_NUM_V_GRID_VALUES;

                // call Quantlib to get the NPV
                QuantlibBinomial(inputData, outputResult, k, numberTests);

                // outout the results to file
                for (int i = 0; i < numberTests; i++) {
#if 0
					std::cout << "[" << i << "]";
					std::cout << " K:" << inputDataVector[i].K;
					std::cout << " rf:" << inputDataVector[i].rf;
					std::cout << " T:" << inputDataVector[i].T;
					std::cout << " q:" << inputDataVector[i].q;
					std::cout << " N:" << inputDataVector[i].N;
					std::cout << " S:" << inputDataVector[i].S;
					std::cout << " V:" << inputDataVector[i].V;
					std::cout << " Result:" << outputResult[i];
					std::cout << std::endl;
#endif

                    // write to file
                    outputFileTestCase << std::fixed << std::setprecision(10) << outputResult[i] << std::endl;
                }

                // close file handle
                outputFileTestCase.close();
            }
        }

    } catch (const std::exception& ex) {
        std::cout << ex.what() << std::endl;
    } catch (const std::string& ex) {
        std::cout << ex << std::endl;
    } catch (...) {
        std::cout << "Exception" << std::endl;
    }

    // close file handle
    inputFileTestCases.close();
    inputFileSVGrid.close();
}
