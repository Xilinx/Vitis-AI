/* -*- mode: c++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 4 -*- */

/*
 Copyright (C) 2003 Ferdinando Ametrano
 Copyright (C) 2006 StatPro Italia srl
 Copyright (C) 2009 Bojan Nikolic

 This file is part of QuantLib, a free-software/open-source library
 for financial quantitative analysts and developers - http://quantlib.org/

 QuantLib is free software: you can redistribute it and/or modify it
 under the terms of the QuantLib license.  You should have received a
 copy of the license along with this program; if not, please email
 <quantlib-dev@lists.sf.net>. The license is also available online at
 <http://quantlib.org/license.shtml>.

 This program is distributed in the hope that it will be useful, but WITHOUT
 ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
 FOR A PARTICULAR PURPOSE.  See the license for more details.
*/

/*! \file brownianbridge.hpp
    \brief Browian bridge
*/

// ===========================================================================
// NOTE: The following copyright notice applies to the original code,
//
// Copyright (C) 2002 Peter J�ckel "Monte Carlo Methods in Finance".
// All rights reserved.
//
// Permission to use, copy, modify, and distribute this software is freely
// granted, provided that this notice is preserved.
// ===========================================================================

/* This file comes from open source QuantLib, https://github.com/lballabio/QuantLib.git. The license for it is BSD-3-Clause.
 * This file is used by XILINX as a reference.
 * In order to simplify its usage, some modification is added to the source file, such as change the data type, remove some not used codes.
 * The modification is done by XILINX.
 */
#ifndef quantlib_brownian_bridge_hpp
#define quantlib_brownian_bridge_hpp

#include <cmath>
#include <vector>

namespace QuantLib {

    class BrownianBridge {

      public:
        BrownianBridge(int steps)
            : size_(steps), t_(size_), sqrtdt_(size_),
              bridgeIndex_(size_), leftIndex_(size_), rightIndex_(size_),
              leftWeight_(size_), rightWeight_(size_), stdDev_(size_) 
        {                  
            for (int i = 0; i < size_; ++i) {
                t_[i] = static_cast<double>(i + 1);
            }
            
            initialize();
        }

        BrownianBridge(const std::vector<double>& times)
             : size_(times.size()), t_(times), sqrtdt_(size_),
               bridgeIndex_(size_), leftIndex_(size_), rightIndex_(size_),
               leftWeight_(size_), rightWeight_(size_), stdDev_(size_) 
        {
            initialize();
        }
        
        // input: begin, end
        void transform(const std::vector<double>& input,
                       std::vector<double>& output) 
        {
            int inputSize = input.size();
            if (inputSize <= 0) {
                std::cout << "empty input sequence" << std::endl;
                return;
            }

            if (inputSize != size_) {
                std::cout << "incompatible sequence size" << std::endl;
                return;
            }

            output.resize(size_);

            output[size_-1] = stdDev_[0] * input[0];

            for (int i = 1; i < size_; ++i) {
                int j = leftIndex_[i];
                int k = rightIndex_[i];
                int l = bridgeIndex_[i];

                if (j != 0) {
                    output[l] =
                        leftWeight_[i] * output[j - 1] +
                        rightWeight_[i] * output[k] +
                        stdDev_[i] * input[i];
                } else {
                    output[l] =
                        rightWeight_[i] * output[k] +
                        stdDev_[i] * input[i];
                }
            }

            for (int i = size_ - 1; i >= 1; --i) {
                output[i] -= output[i-1];
                output[i] /= sqrtdt_[i];
            }

            output[0] /= sqrtdt_[0];

        }

      private:
        void initialize() {

            sqrtdt_[0] = std::sqrt(t_[0]);

            for (int i = 1; i < size_; ++i) {
                sqrtdt_[i] = std::sqrt(t_[i] - t_[i - 1]);
            }

            std::vector<int> map(size_, 0);

            map[size_-1] = 1;

            bridgeIndex_[0] = size_-1;

            stdDev_[0] = std::sqrt(t_[size_-1]);

            leftWeight_[0] = rightWeight_[0] = 0.0;

            for (int j = 0, i = 1; i < size_; ++i) {
                while (map[j]) {
                    ++j;
                }
                
                int k = j;

                while (!map[k]) {
                    ++k;
                }

                int l = j + ((k - 1 - j) >> 1);
                
                map[l] = i;

                bridgeIndex_[i] = l;
                leftIndex_[i]   = j;
                rightIndex_[i]  = k;
                #if 0
                std::cout<<"l = "<<l<<", j="<<j<<", k="<<k<<std::endl;
                #endif

                if (j != 0) {
                    leftWeight_[i] = (t_[k]-t_[l])/(t_[k]-t_[j-1]);
                    rightWeight_[i] = (t_[l]-t_[j-1])/(t_[k]-t_[j-1]);
                    stdDev_[i] =
                        std::sqrt(((t_[l]-t_[j-1])*(t_[k]-t_[l]))
                                /(t_[k]-t_[j-1]));
                } else {
                    leftWeight_[i]  = (t_[k]-t_[l])/t_[k];
                    rightWeight_[i] =  t_[l]/t_[k];
                    stdDev_[i] = std::sqrt(t_[l]*(t_[k]-t_[l])/t_[k]);
                }

                j = k + 1;

                if (j >= size_) {
                    j = 0;    //  wrap around
                }
            }
            #if 0
            for(int i = 0; i < size_; ++i) {
                std::cout<<"i = "<<i<<", leftWeight_="<<leftWeight_[i]<<", rightWeight_="<<rightWeight_[i]<<", stdDev_="<<stdDev_[i]<<std::endl;
                std::cout<<"i = "<<i<<", bridgeIndex_="<<bridgeIndex_[i]<<", leftIndex_="<<leftIndex_[i]<<", rightIndex_="<<rightIndex_[i]<<std::endl;
                std::cout<<"i = "<<i<<", t_="<<t_[i]<<", sqrtdt_="<<sqrtdt_[i]<<std::endl;
            }
            #endif
        }

      private:
        int size_;
        std::vector<double> t_;
        std::vector<double> sqrtdt_;
        std::vector<int> bridgeIndex_, leftIndex_, rightIndex_;
        std::vector<double> leftWeight_, rightWeight_, stdDev_;
    };

}

#endif
