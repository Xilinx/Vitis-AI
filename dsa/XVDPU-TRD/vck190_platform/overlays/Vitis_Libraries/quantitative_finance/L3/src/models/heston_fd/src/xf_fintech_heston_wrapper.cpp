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
/*--
 * ---------------------------------------------------------------------------------------------------------------------*/

#include <chrono>
#include <vector>

#include "xf_fintech_api.hpp"
#include "xf_fintech_error_codes.hpp"
#include "xf_fintech_internal.hpp"
#include "xf_fintech_trace.hpp"
#include "xf_fintech_types.hpp"

#include "xf_fintech_heston_kernel_constants.hpp"

#include "xf_fintech_heston.hpp"
#include "xf_fintech_heston_ocl_objects.hpp"
#include "xf_fintech_li.hpp"

using namespace xf::fintech;

typedef struct _XCLBINLookupElement {
    Device::DeviceType deviceType;
    std::string xclbinName;

} XCLBINLookupElement;

static XCLBINLookupElement XCLBIN_LOOKUP_TABLE[] = {
    {Device::DeviceType::U50, "fd_heston_kernel_u50_hw_m8192_double.xclbin"},
    {Device::DeviceType::U200, "fd_heston_kernel_u200_hw_m8192_double.xclbin"},
    {Device::DeviceType::U250, "fd_heston_kernel_u250_hw_m8192_double.xclbin"},
    {Device::DeviceType::U280, "fd_heston_kernel_u280_hw_m8192_double.xclbin"}};

static const unsigned int NUM_XCLBIN_LOOKUP_TABLE_ENTRIES =
    sizeof(XCLBIN_LOOKUP_TABLE) / sizeof(XCLBIN_LOOKUP_TABLE[0]);

// FDHeston::FDHeston() : FDHeston(DEFAULT_M1, DEFAULT_M2) {}

FDHeston::FDHeston(int M1, int M2, std::string xclbin) {
    m_pContext = nullptr;
    m_pCommandQueue = nullptr;
    m_pProgram = nullptr;

    m_M1 = M1;
    m_M2 = M2;
    m_xclbin = xclbin;
}

FDHeston::~FDHeston() {
    if (deviceIsPrepared()) {
        releaseDevice();
    }
}

std::string FDHeston::getXCLBINName(Device* device) {
    return m_xclbin;
}

int FDHeston::createOCLObjects(Device* device) {
    int retval = XLNX_OK;
    cl_int cl_retval = CL_SUCCESS;

    std::chrono::time_point<std::chrono::high_resolution_clock> start;
    std::chrono::time_point<std::chrono::high_resolution_clock> end;

    std::string xclbinName;

    cl::Device clDevice;

    clDevice = device->getCLDevice();

    m_pContext = new cl::Context(clDevice, nullptr, nullptr, nullptr, &cl_retval);

    if (cl_retval == CL_SUCCESS) {
        m_pCommandQueue = new cl::CommandQueue(*m_pContext, clDevice, CL_QUEUE_PROFILING_ENABLE, &cl_retval);
    }

    if (cl_retval == CL_SUCCESS) {
        start = std::chrono::high_resolution_clock::now();

        xclbinName = getXCLBINName(device);

        m_binaries.clear();
        m_binaries = xcl::import_binary_file(xclbinName);

        end = std::chrono::high_resolution_clock::now();

        Trace::printInfo("[XLNX] Binary Import Time = %lld microseconds\n",
                         std::chrono::duration_cast<std::chrono::microseconds>(end - start).count());
    }

    /////////////////////////
    // Create PROGRAM Object
    /////////////////////////
    if (cl_retval == CL_SUCCESS) {
        std::vector<cl::Device> devicesToProgram;
        devicesToProgram.push_back(clDevice);

        start = std::chrono::high_resolution_clock::now();

        m_pProgram = new cl::Program(*m_pContext, devicesToProgram, m_binaries, nullptr, &cl_retval);

        end = std::chrono::high_resolution_clock::now();

        Trace::printInfo("[XLNX] Device Programming Time = %lld microseconds\n",
                         std::chrono::duration_cast<std::chrono::microseconds>(end - start).count());
    }

    /////////////////////////
    // Create KERNEL Objects
    /////////////////////////
    if (cl_retval == CL_SUCCESS) {
        m_pKernel = new cl::Kernel(*m_pProgram, "fd_kernel", &cl_retval);
    }

    return retval;
}

int FDHeston::releaseOCLObjects(void) {
    int retval = XLNX_OK;
    unsigned int i;

    if (m_pKernel != nullptr) {
        delete (m_pKernel);
        m_pKernel = nullptr;
    }

    if (m_pProgram != nullptr) {
        delete (m_pProgram);
        m_pProgram = nullptr;
    }

    for (i = 0; i < m_binaries.size(); i++) {
        std::pair<const void*, cl::size_type> binaryPair = m_binaries[i];
        delete[](char*)(binaryPair.first);
    }

    if (m_pCommandQueue != nullptr) {
        delete (m_pCommandQueue);
        m_pCommandQueue = nullptr;
    }

    if (m_pContext != nullptr) {
        delete (m_pContext);
        m_pContext = nullptr;
    }

    return retval;
}

int FDHeston::run(double stockPrice,
                  double strikePrice,
                  double riskFreeRateDomestic,
                  double volatility,
                  double timeToMaturity,
                  double meanReversionRate,      // kappa
                  double volatilityOfVolatility, // sig
                  double correlationCoefficient, // rho
                  double longRunAveragePrice,    // eta
                  double* pOptionPrice) {
    int retval = XLNX_OK;

    // NOTE - run timers are handled by internal function...

    retval = this->run(stockPrice, strikePrice, riskFreeRateDomestic, volatility, timeToMaturity, meanReversionRate,
                       volatilityOfVolatility, correlationCoefficient, longRunAveragePrice, DEFAULT_N, pOptionPrice);

    return retval;
}

int FDHeston::run(double stockPrice,
                  double strikePrice,
                  double riskFreeRateDomestic,
                  double volatility,
                  double timeToMaturity,
                  double meanReversionRate,      // kappa
                  double volatilityOfVolatility, // sigma
                  double correlationCoefficient, // rho
                  double longRunAveragePrice,    // eta
                  int numSteps,
                  double* pOptionPrice) {
    int retval = XLNX_OK;
    HestonFD::HestonFDReturnVal hestonRetVal = HestonFD::HestonFDReturnVal::XLNXOK;

    int m1;
    int m2;

    m_runStartTime = std::chrono::high_resolution_clock::now();

    if (deviceIsPrepared()) {
        HestonFDOCLObjects oclObjects(m_pContext, m_pCommandQueue, m_pKernel);

        // Pass in solver and model parameters
        HestonFDModelParameters model_parameters(strikePrice, stockPrice, volatility, timeToMaturity, meanReversionRate,
                                                 volatilityOfVolatility, correlationCoefficient, longRunAveragePrice,
                                                 riskFreeRateDomestic, DEFAULT_RF);

        HestonFDSolverParameters solver_parameters(model_parameters);

        solver_parameters.Set_m1(m_M1);
        solver_parameters.Set_m2(m_M2);
        solver_parameters.Set_N(numSteps);

        // Create memory for results
        HestonFDPriceRam price_ram(solver_parameters);
        HestonFD heston(model_parameters, solver_parameters, oclObjects);

        // Solve

        std::vector<double> s_grid;
        std::vector<double> v_grid;

        hestonRetVal = heston.Solve(price_ram, s_grid, v_grid);

        if (hestonRetVal != HestonFD::HestonFDReturnVal::XLNXOK) {
            retval = XLNX_ERROR_MODEL_INTERNAL_ERROR;
        }

        if (retval == XLNX_OK) {
            m1 = solver_parameters.Get_m1();
            m2 = solver_parameters.Get_m2();

            if (!Xilinx_Interpolate(price_ram.Get_PriceGrid(), s_grid.data(), v_grid.data(), m1, m2, stockPrice,
                                    volatility, pOptionPrice)) {
                Trace::printError("[XLNX] ERROR: failed to calculate the NPV\n");
                retval = XLNX_ERROR_LINEAR_INTERPOLATION_FAILED;
            } else {
                Trace::printInfo("[XLNX] NPV: %f\n", *pOptionPrice);
            }
        }

    } else {
        retval = XLNX_ERROR_DEVICE_NOT_OWNED_BY_SPECIFIED_OCL_CONTROLLER;
    }

    m_runEndTime = std::chrono::high_resolution_clock::now();

    return retval;
}

int FDHeston::run(double stockPrice,
                  double strikePrice,
                  double riskFreeRateDomestic,
                  double volatility,
                  double timeToMaturity,
                  double meanReversionRate,      // kappa
                  double volatilityOfVolatility, // sigma
                  double correlationCoefficient, // rho
                  double longRunAveragePrice,    // eta
                  double* pOptionPrice,
                  double* pDelta,
                  double* pVega,
                  double* pGamma,
                  double* pVolga,
                  double* pVanna) {
    int retval = XLNX_OK;

    // NOTE - run timers are handled by internal function...

    retval = this->run(stockPrice, strikePrice, riskFreeRateDomestic, volatility, timeToMaturity, meanReversionRate,
                       volatilityOfVolatility, correlationCoefficient, longRunAveragePrice, DEFAULT_N, pOptionPrice,
                       pDelta, pVega, pGamma, pVolga, pVanna);

    return retval;
}

int FDHeston::run(double stockPrice,
                  double strikePrice,
                  double riskFreeRateDomestic,
                  double volatility,
                  double timeToMaturity,
                  double meanReversionRate,      // kappa
                  double volatilityOfVolatility, // sigma
                  double correlationCoefficient, // rho
                  double longRunAveragePrice,    // eta
                  int numSteps,
                  double* pOptionPrice,
                  double* pDelta,
                  double* pVega,
                  double* pGamma,
                  double* pVolga,
                  double* pVanna) {
    int retval = XLNX_OK;
    HestonFD::HestonFDReturnVal hestonRetVal = HestonFD::HestonFDReturnVal::XLNXOK;

    int m1;
    int m2;

    m_runStartTime = std::chrono::high_resolution_clock::now();

    if (deviceIsPrepared()) {
        HestonFDOCLObjects oclObjects(m_pContext, m_pCommandQueue, m_pKernel);

        // Pass in solver and model parameters
        HestonFDModelParameters model_parameters(strikePrice, stockPrice, volatility, timeToMaturity, meanReversionRate,
                                                 volatilityOfVolatility, correlationCoefficient, longRunAveragePrice,
                                                 riskFreeRateDomestic, DEFAULT_RF);

        HestonFDSolverParameters solver_parameters(model_parameters);

        solver_parameters.Set_m1(m_M1);
        solver_parameters.Set_m2(m_M2);
        solver_parameters.Set_N(numSteps);

        // Create memory for results
        HestonFDPriceRam price_ram(solver_parameters);
        HestonFD heston(model_parameters, solver_parameters, oclObjects);

        // Solve

        std::vector<double> s_grid;
        std::vector<double> v_grid;

        hestonRetVal = heston.Solve(price_ram, s_grid, v_grid);

        if (hestonRetVal != HestonFD::HestonFDReturnVal::XLNXOK) {
            retval = XLNX_ERROR_MODEL_INTERNAL_ERROR;
        }

        if (retval == XLNX_OK) {
            m1 = solver_parameters.Get_m1();
            m2 = solver_parameters.Get_m2();

            if (!Xilinx_Interpolate(price_ram.Get_PriceGrid(), s_grid.data(), v_grid.data(), m1, m2, stockPrice,
                                    volatility, pOptionPrice)) {
                Trace::printError("[XLNX] ERROR: failed to calculate the NPV\n");
                retval = XLNX_ERROR_LINEAR_INTERPOLATION_FAILED;
            } else {
                Trace::printInfo("[XLNX] NPV: %f\n", *pOptionPrice);
            }
        }

        if (retval == XLNX_OK) {
            vector<double> priceGrid;
            double* priceGridData = price_ram.Get_PriceGrid();
            int priceGridSize = price_ram.Get_m1_GridSize() * price_ram.Get_m2_GridSize();

            for (int i = 0; i < priceGridSize; i++) {
                priceGrid.push_back(priceGridData[i]);
            }

            retval = calculateGreeksMinimalGrid(priceGrid, s_grid, v_grid, stockPrice, volatility, *pDelta, *pVega,
                                                *pGamma, *pVolga, *pVanna);
        }
    } else {
        retval = XLNX_ERROR_DEVICE_NOT_OWNED_BY_SPECIFIED_OCL_CONTROLLER;
    }

    m_runEndTime = std::chrono::high_resolution_clock::now();

    return retval;
}

int FDHeston::run(double stockPrice,
                  double strikePrice,
                  double riskFreeRateDomestic,
                  double volatility,
                  double timeToMaturity,
                  double meanReversionRate,      // kappa
                  double volatilityOfVolatility, // sigma
                  double correlationCoefficient, // rho
                  double longRunAveragePrice,    // eta
                  int numSteps,
                  std::vector<double>& priceGrid,
                  std::vector<double>& sGrid,
                  std::vector<double>& vGrid) {
    int retval = XLNX_OK;

    HestonFD::HestonFDReturnVal hestonRetVal = HestonFD::HestonFDReturnVal::XLNXOK;

    m_runStartTime = std::chrono::high_resolution_clock::now();

    if (deviceIsPrepared()) {
        HestonFDOCLObjects oclObjects(m_pContext, m_pCommandQueue, m_pKernel);

        // Pass in solver and model parameters
        HestonFDModelParameters model_parameters(strikePrice, stockPrice, volatility, timeToMaturity, meanReversionRate,
                                                 volatilityOfVolatility, correlationCoefficient, longRunAveragePrice,
                                                 riskFreeRateDomestic, DEFAULT_RF);

        HestonFDSolverParameters solver_parameters(model_parameters);

        solver_parameters.Set_m1(m_M1);
        solver_parameters.Set_m2(m_M2);
        solver_parameters.Set_N(numSteps);

        // Create memory for results
        HestonFDPriceRam price_ram(solver_parameters);
        HestonFD heston(model_parameters, solver_parameters, oclObjects);

        // Solve
        priceGrid.clear();
        sGrid.clear();
        vGrid.clear();

        hestonRetVal = heston.Solve(price_ram, sGrid, vGrid);

        if (hestonRetVal != HestonFD::HestonFDReturnVal::XLNXOK) {
            retval = XLNX_ERROR_MODEL_INTERNAL_ERROR;
        }

        if (retval == XLNX_OK) {
            // give back the price data
            double* priceGridData = price_ram.Get_PriceGrid();
            int priceGridSize = price_ram.Get_m1_GridSize() * price_ram.Get_m2_GridSize();

            for (int i = 0; i < priceGridSize; i++) {
                priceGrid.push_back(priceGridData[i]);
            }
        }
    } else {
        retval = XLNX_ERROR_DEVICE_NOT_OWNED_BY_SPECIFIED_OCL_CONTROLLER;
    }

    m_runEndTime = std::chrono::high_resolution_clock::now();

    return retval;
}

long long int FDHeston::getLastRunTime(void) {
    long long int duration = 0;

    duration =
        (long long int)std::chrono::duration_cast<std::chrono::microseconds>(m_runEndTime - m_runStartTime).count();

    return duration;
}

/****************************************************************************************************************/

void FDHeston::calculateGradient2D(std::vector<double>& inputData2D,
                                   std::vector<double>& rowPositions,
                                   std::vector<double>& columnPositions,
                                   std::vector<double>& outputGradient1,
                                   std::vector<double>& outputGradient2) {
    std::vector<double> rowSpacing;
    std::vector<double> columnSpacing;

    int numRows = rowPositions.size();
    int numColumns = columnPositions.size();

    double dx1;
    double dx2;

    double a;
    double b;
    double c;

    outputGradient1.resize(inputData2D.size());
    outputGradient2.resize(inputData2D.size());

    // calculate the SPACING between the N'th and N+1'th elements...
    for (int i = 0; i < numRows - 1; i++) {
        rowSpacing.push_back(rowPositions[i + 1] - rowPositions[i]);
    }

    for (int i = 0; i < numColumns - 1; i++) {
        columnSpacing.push_back(columnPositions[i + 1] - columnPositions[i]);
    }

    //
    // Calculate the gradient in one direction....
    //
    for (int row = 0; row < numRows; row++) {
        for (int column = 0; column < numColumns; column++) {
            if (column == 0) // edge condition
            {
                dx1 = columnSpacing[0];

                outputGradient1[(row * numColumns) + column] =
                    (inputData2D[(row * numColumns) + column + 1] - inputData2D[(row * numColumns) + column]) / dx1;
            } else if (column == numColumns - 1) // other edge condition
            {
                dx1 = columnSpacing[columnSpacing.size() - 1];

                outputGradient1[(row * numColumns) + column] =
                    (inputData2D[(row * numColumns) + column] - inputData2D[(row * numColumns) + (column - 1)]) / dx1;

            } else {
                dx1 = columnSpacing[column - 1];
                dx2 = columnSpacing[column];

                a = -(dx2) / (dx1 * (dx1 + dx2));
                b = (dx2 - dx1) / (dx1 * dx2);
                c = dx1 / (dx2 * (dx1 + dx2));

                outputGradient1[(row * numColumns) + column] = (a * inputData2D[(row * numColumns) + (column - 1)]) +
                                                               (b * inputData2D[(row * numColumns) + (column)]) +
                                                               (c * inputData2D[(row * numColumns) + (column + 1)]);
            }
        }
    }

    //
    // ... and calculate the gradient in the other direction...
    //
    for (int row = 0; row < numRows; row++) {
        for (int column = 0; column < numColumns; column++) {
            if (row == 0) // edge condition
            {
                dx1 = rowSpacing[0];

                outputGradient2[(row * numColumns) + column] =
                    (inputData2D[((row + 1) * numColumns) + column] - inputData2D[(row * numColumns) + column]) / dx1;
            } else if (row == numRows - 1) // other edge condition
            {
                dx1 = rowSpacing[rowSpacing.size() - 1];

                outputGradient2[(row * numColumns) + column] =
                    (inputData2D[(row * numColumns) + column] - inputData2D[((row - 1) * numColumns) + column]) / dx1;

            } else {
                dx1 = rowSpacing[row - 1];
                dx2 = rowSpacing[row];

                a = -(dx2) / (dx1 * (dx1 + dx2));
                b = (dx2 - dx1) / (dx1 * dx2);
                c = dx1 / (dx2 * (dx1 + dx2));

                outputGradient2[(row * numColumns) + column] = (a * inputData2D[((row - 1) * numColumns) + column]) +
                                                               (b * inputData2D[(row * numColumns) + column]) +
                                                               (c * inputData2D[((row + 1) * numColumns) + column]);
            }
        }
    }
}

int FDHeston::calculateGreeks(std::vector<double>& priceGrid,
                              std::vector<double>& S,
                              std::vector<double>& V,
                              double stockPrice,
                              double volatility,
                              double& outputDelta,
                              double& outputVega,
                              double& outputGamma,
                              double& outputVolga,
                              double& outputVanna) {
    int retval = XLNX_OK;

    std::vector<double> grad1;
    std::vector<double> grad2;

    calculateGradient2D(priceGrid, V, S, grad1, grad2);

    std::vector<double> d2u_ds2;
    std::vector<double> d2u_dsdv;

    calculateGradient2D(grad1, V, S, d2u_ds2, d2u_dsdv);

    std::vector<double> d2u_dvds;
    std::vector<double> d2u_dvs;

    calculateGradient2D(grad2, V, S, d2u_dvds, d2u_dvs);

    // Delta is du/ds.  This takes the ds/du grid - grad1 - and interpolates at a fixed point.
    if (!Xilinx_Interpolate(grad1.data(), S.data(), V.data(), S.size(), V.size(), stockPrice, volatility,
                            &outputDelta)) {
        retval = XLNX_ERROR_LINEAR_INTERPOLATION_FAILED;
    }

    if (retval == XLNX_OK) {
        // Vega has a few definitions, we get Vega1 from the Rouah book p328
        // Vega1 is du / dv.This takes the ds / dv grid - grad2 - and interpolates at a fixed point.
        if (!Xilinx_Interpolate(grad2.data(), S.data(), V.data(), S.size(), V.size(), stockPrice, volatility,
                                &outputVega)) {
            retval = XLNX_ERROR_LINEAR_INTERPOLATION_FAILED;
        }
    }

    if (retval == XLNX_OK) {
        // Gamma is d2u/ds2 - use first element second differential grid
        if (!Xilinx_Interpolate(d2u_ds2.data(), S.data(), V.data(), S.size(), V.size(), stockPrice, volatility,
                                &outputGamma)) {
            retval = XLNX_ERROR_LINEAR_INTERPOLATION_FAILED;
        }
    }

    if (retval == XLNX_OK) {
        // Volga is d2u / dv2 - use second element second differential grid
        if (!Xilinx_Interpolate(d2u_dvs.data(), S.data(), V.data(), S.size(), V.size(), stockPrice, volatility,
                                &outputVolga)) {
            retval = XLNX_ERROR_LINEAR_INTERPOLATION_FAILED;
        }
    }

    if (retval == XLNX_OK) {
        // Vanna is d2u / dvds - use first element second differential grid
        // if(!Xilinx_Interpolate(d2u_dvds.data(), S.data(), V.data(), S.size(), V.size(), stockPrice, volatility,
        // &outputVanna))
        if (!Xilinx_Interpolate(d2u_dsdv.data(), S.data(), V.data(), S.size(), V.size(), stockPrice, volatility,
                                &outputVanna)) {
            retval = XLNX_ERROR_LINEAR_INTERPOLATION_FAILED;
        }
    }

    return retval;
}

int FDHeston::calculateGreeksMinimalGrid(std::vector<double>& priceGrid,
                                         std::vector<double>& S,
                                         std::vector<double>& V,
                                         double stockPrice,
                                         double volatility,
                                         double& outputDelta,
                                         double& outputVega,
                                         double& outputGamma,
                                         double& outputVolga,
                                         double& outputVanna) {
    int retval = XLNX_OK;

    std::vector<double> miniPriceGrid;
    std::vector<double> miniS;
    std::vector<double> miniV;

    bool bFoundExactS = false;
    int SExactIndex = 0;
    int SUpperIndex = 0;
    int SLowerIndex = 0;

    bool bFoundExactV = false;
    int VExactIndex = 0;
    int VUpperIndex = 0;
    int VLowerIndex = 0;

    int row;
    int column;

    // int numRows = (int)V.size(); //currently unused
    int numColumns = (int)S.size();

    //
    // Locate the indexes of the S and V vectors that contain the exact (or nearest) values to the requested stockprice
    // and volatility
    //
    for (int i = 0; i < (int)S.size(); i++) {
        if (S[i] >= stockPrice) {
            if (S[i] == stockPrice) {
                bFoundExactS = true;
                SExactIndex = i;
            } else {
                SUpperIndex = i;
                SLowerIndex = SUpperIndex - 1;
            }
            break;
        }
    }

    for (int i = 0; i < (int)V.size(); i++) {
        if (V[i] >= volatility) {
            if (V[i] == volatility) {
                bFoundExactV = true;
                VExactIndex = i;
            } else {
                VUpperIndex = i;
                VLowerIndex = VUpperIndex - 1;
            }

            break; // out of loop
        }
    }

    //
    // figure out the upper and lower indexes of the main price grid to form our mini-grid
    //
    if (bFoundExactS) {
        SUpperIndex = SExactIndex + 2;
        SLowerIndex = SExactIndex - 2;
    } else {
        SUpperIndex = SUpperIndex + 2;
        SLowerIndex = SLowerIndex - 2;
    }

    if (bFoundExactV) {
        VUpperIndex = VExactIndex + 2;
        VLowerIndex = VExactIndex - 2;
    } else {
        VUpperIndex = VUpperIndex + 2;
        VLowerIndex = VLowerIndex - 2;
    }

    //
    // Make sure to limit our upper/lower indexes to be valid within the given vectors....
    //
    if (SUpperIndex >= (int)S.size()) {
        SUpperIndex = (int)S.size() - 1;
    }

    if (SLowerIndex < 0) {
        SLowerIndex = 0;
    }

    if (VUpperIndex >= (int)V.size()) {
        VUpperIndex = (int)V.size() - 1;
    }

    if (VLowerIndex < 0) {
        VLowerIndex = 0;
    }

    //
    // Create the mini-vectors of S and V data...contain only the minimal set of values we need...
    //
    for (column = SLowerIndex; column <= SUpperIndex; column++) {
        miniS.push_back(S[column]);
    }

    for (row = VLowerIndex; row <= VUpperIndex; row++) {
        miniV.push_back(V[row]);
    }

    //
    // Create the mini price-grid.
    //
    for (row = VLowerIndex; row <= VUpperIndex; row++) {
        for (column = SLowerIndex; column <= SUpperIndex; column++) {
            miniPriceGrid.push_back(priceGrid[(row * numColumns) + column]);
        }
    }

    //
    // Call the processing method, passing our mini-data...
    //
    retval = calculateGreeks(miniPriceGrid, miniS, miniV, stockPrice, volatility, outputDelta, outputVega, outputGamma,
                             outputVolga, outputVanna);

    return retval;
}
