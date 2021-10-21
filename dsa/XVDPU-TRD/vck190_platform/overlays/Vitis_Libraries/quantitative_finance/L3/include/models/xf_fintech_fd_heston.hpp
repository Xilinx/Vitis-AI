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

#ifndef _XF_FINTECH_HESTON_FD_H_
#define _XF_FINTECH_HESTON_FD_H_

#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "xf_fintech_device.hpp"
#include "xf_fintech_ocl_controller.hpp"
#include "xf_fintech_types.hpp"

namespace xf {
namespace fintech {

/**
 * @class FDHeston
 *
 * @brief This class implements the Finite Difference Heston Model.
 *
 * It is intended that the user will populate the asset data when calling the run() method.
 * A number of overriden methods have been provided that allow configuration of the number
 * of steps and return of the Greeks.
 */

class FDHeston : public OCLController {
   public:
    FDHeston();
    FDHeston(int M1, int M2, std::string xclbin);

   public:
    virtual ~FDHeston();

   public:
    /**
     * Calculate a single option price with default number of steps.
     *
     * @param stockPrice the stock price
     * @param strikePrice the strike price
     * @param riskFreeRateDomestic the risk free domestic interest rate
     * @param volatility the volatility
     * @param timeToMaturity the time to maturity
     * @param meanReversionRate the mean reversion rate (kappa)
     * @param volatilityOfVolatility the volatility of volatility (sigma)
     * @param correlationCoefficient the correlation coefficient (rho)
     * @param longRunAveragePrice the returned option price (eta)
     * @param pOptionPrice the returned option price
     *
     */
    int run(double stockPrice,
            double strikePrice,
            double riskFreeRateDomestic,
            double volatility,
            double timeToMaturity,
            double meanReversionRate,
            double volatilityOfVolatility,
            double correlationCoefficient,
            double longRunAveragePrice,
            double* pOptionPrice);

    /**
     * Calculate a single option price with a configurable number of steps.
     *
     * @param stockPrice the stock price
     * @param strikePrice the strike price
     * @param riskFreeRateDomestic the risk free domestic interest rate
     * @param volatility the volatility
     * @param timeToMaturity the time to maturity
     * @param meanReversionRate the mean reversion rate (kappa)
     * @param volatilityOfVolatility the volatility of volatility (sigma)
     * @param correlationCoefficient the correlation coefficient (rho)
     * @param longRunAveragePrice the returned option price (eta)
     * @param numSteps the number of steps
     * @param pOptionPrice the returned option price
     *
     */
    int run(double stockPrice,
            double strikePrice,
            double riskFreeRateDomestic,
            double volatility,
            double timeToMaturity,
            double meanReversionRate,
            double volatilityOfVolatility,
            double correlationCoefficient,
            double longRunAveragePrice,
            int numSteps,
            double* pOptionPrice);

    /**
     * Calculate a single option price with a default number of steps and return the greeks.
     *
     * @param stockPrice the stock price
     * @param strikePrice the strike price
     * @param riskFreeRateDomestic the risk free domestic interest rate
     * @param volatility the volatility
     * @param timeToMaturity the time to maturity
     * @param meanReversionRate the mean reversion rate (kappa)
     * @param volatilityOfVolatility the volatility of volatility (sigma)
     * @param correlationCoefficient the correlation coefficient (rho)
     * @param longRunAveragePrice the returned option price (eta)
     * @param pOptionPrice the returned option price
     * @param pDelta the returned greek Delta
     * @param pVega the returned greek Vega
     * @param pGamma the returned greek Gamma
     * @param pVolga the returned greek Volga
     * @param pVanna the returned greek Vanna
     *
     */
    int run(double stockPrice,
            double strikePrice,
            double riskFreeRateDomestic,
            double volatility,
            double timeToMaturity,
            double meanReversionRate,
            double volatilityOfVolatility,
            double correlationCoefficient,
            double longRunAveragePrice,
            double* pOptionPrice,
            double* pDelta,
            double* pVega,
            double* pGamma,
            double* pVolga,
            double* pVanna);

    /**
     * Calculate a single option price with a configurable number of steps and return the greeks.
     *
     * @param stockPrice the stock price
     * @param strikePrice the strike price
     * @param riskFreeRateDomestic the risk free domestic interest rate
     * @param volatility the volatility
     * @param timeToMaturity the time to maturity
     * @param meanReversionRate the mean reversion rate (kappa)
     * @param volatilityOfVolatility the volatility of volatility (sigma)
     * @param correlationCoefficient the correlation coefficient (rho)
     * @param longRunAveragePrice the returned option price (eta)
     * @param numSteps the number of steps
     * @param pOptionPrice the returned option price
     * @param pDelta the returned greek Delta
     * @param pVega the returned greek Vega
     * @param pGamma the returned greek Gamma
     * @param pVolga the returned greek Volga
     * @param pVanna the returned greek Vanna
     */
    int run(double stockPrice,
            double strikePrice,
            double riskFreeRateDomestic,
            double volatility,
            double timeToMaturity,
            double meanReversionRate,
            double volatilityOfVolatility,
            double correlationCoefficient,
            double longRunAveragePrice,
            int numSteps,
            double* pOptionPrice,
            double* pDelta,
            double* pVega,
            double* pGamma,
            double* pVolga,
            double* pVanna);

    // The following interface is intended for Xilinx internal use only.
    int run(double stockPrice,
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
            std::vector<double>& vGrid);

   public:
    /**
     * This method returns the time the execution of the last call to run() took.
     */
    long long int getLastRunTime(void); // in microseconds

   private:
    // OCLController interface
    int createOCLObjects(Device* device);
    int releaseOCLObjects(void);
    std::string getXCLBINName(Device* device);

    cl::Context* m_pContext;
    cl::CommandQueue* m_pCommandQueue;
    cl::Program::Binaries m_binaries;
    cl::Program* m_pProgram;
    cl::Kernel* m_pKernel;
    std::chrono::time_point<std::chrono::high_resolution_clock> m_runStartTime;
    std::chrono::time_point<std::chrono::high_resolution_clock> m_runEndTime;

    int m_M1;
    int m_M2;
    std::string m_xclbin;

    static const int DEFAULT_M1 = 128;
    static const int DEFAULT_M2 = 64;
    static const int DEFAULT_N = 200;

    static constexpr double DEFAULT_RF = 0; // JIRA: DCA-133

    // This method takes in the full price grid (default 128x64), then creates a mini-grid (and mini S and V vectors)
    // containing just enough cells to correctly
    // calculate the greeks for a single NPV.
    // It then calls the "calculateGreeks" method, passing in the mini-data.
    int calculateGreeksMinimalGrid(std::vector<double>& priceGrid,
                                   std::vector<double>& S,
                                   std::vector<double>& V,
                                   double stockPrice,
                                   double volatility,
                                   double& outputDelta,
                                   double& outputVega,
                                   double& outputGamma,
                                   double& outputVolga,
                                   double& outputVanna);

    int calculateGreeks(std::vector<double>& priceGrid,
                        std::vector<double>& S,
                        std::vector<double>& V,
                        double stockPrice,
                        double volatility,
                        double& outputDelta,
                        double& outputVega,
                        double& outputGamma,
                        double& outputVolga,
                        double& outputVanna);

    void calculateGradient2D(std::vector<double>& inputData2D,
                             std::vector<double>& rowPositions,
                             std::vector<double>& columnPositions,
                             std::vector<double>& ouptutGradient1,
                             std::vector<double>& outputGradient2);
};

} // end namespace fintech
} // end namespace xf

#endif /* _XF_FINTECH_HESTON_FD_H_ */
