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

/**
 * @file tree_engine.hpp
 *
 * @brief the file include 4 function that are treeSwaptionEngine, treeSwapEngine, treeCapFloorEngine,
 * treeCallableEngine.
 */
#ifndef _XF_FINTECH_TREE_ENGINE_HPP_
#define _XF_FINTECH_TREE_ENGINE_HPP_

#include <ap_fixed.h>
#include <ap_int.h>
#include "xf_fintech/tree_lattice.hpp"
#include "xf_fintech/time_grid.hpp"
#include "xf_fintech/tree_instrument.hpp"
#include "hls_math.h"

#ifndef __SYNTHESIS__
#include <iostream>
#endif

namespace xf {

namespace fintech {

/**
 * @brief Tree Swaption Pricing Engine using Trinomial Tree based 1D Lattice method.
 *
 * @tparam DT supported data type including double and float data type, which decides the precision of result.
 * @tparam Model short-rate model class
 * @tparam Process stochastic process class
 * @tparam DIM 1D or 2D short-rate model
 * @tparam LEN maximum length of timestep, which affects the latency and resources utilization.
 * @tparam LEN2 maximum length of node of tree, which affects the latency and resources utilization.
 *
 * @param model short-rate model that has been initialized
 * @param process parameters of stochastic process
 * @param type 0: Payer, 1: Receiver
 * @param fixedRate fixed annual interest rate.
 * @param timestep estimate the number of discrete steps from 0 to T, T is the maturity time.
 * @param initTime the time including begin timepoint, end timepoint, exercise timepoints, floating coupon timepoints,
 * and fixed coupon timepoints is arranged from small to large. The timepoints are relative values based on the
 * reference date the unit is year.
 * @param initSize the length of array initTime.
 * @param exerciseCnt exercise timepoints count in initTime.
 * @param floatingCnt floating coupon timepoints count in initTime.
 * @param fixedCnt fixed coupon timepoints count in initTime.
 * @param flatRate floating benchmark annual interest rate
 * @param nominal nominal principal
 * @param x0 initial underlying
 * @param spread spreads on interest rates
 * @param NPV is pricing result array of this engine
 */
template <typename DT, typename Model, typename Process, int DIM, int LEN, int LEN2>
void treeSwaptionEngine(Model& model,
                        DT* process,
                        int type,
                        DT fixedRate,
                        int timestep,
                        DT initTime[LEN],
                        int initSize,
                        int* exerciseCnt,
                        int* floatingCnt,
                        int* fixedCnt,
                        DT flatRate,
                        DT nominal,
                        DT x0,
                        DT spread,
                        DT* NPV) {
    DT time[LEN];
    DT dtime[LEN];
    int beginCnt;
    int endCnt;
    int exerciseEndCnt;
    int floatingEndCnt;
    int fixedEndCnt;

    internal::TimeGrid<DT, LEN> grid;

    DT dtMax = initTime[initSize - 1] / timestep;
    grid.calcuGrid(initSize, initTime, dtMax, time, dtime, exerciseEndCnt, exerciseCnt, fixedEndCnt, floatingEndCnt,
                   fixedCnt, floatingCnt, endCnt);

#ifndef __SYNTHESIS__
    cout << "set timesteps=" << timestep << ",actual timesteps=" << endCnt + 1 << endl;
#endif

    // define TreeLattice as lattice
    DT fixedCoupon = nominal * fixedRate;
    DT accruedSpread = 0.0; // nominal * T * spread;

    xf::fintech::TreeLattice<DT, Model, Process, internal::TreeInstrument<DT, 0, LEN2>, DIM, LEN, LEN2> lattice;
    lattice.setup(model, process, endCnt + 1, flatRate, x0, time, dtime);

    internal::TreeInstrument<DT, 0, LEN2> engine;
    engine.initialize(type, nominal, accruedSpread, fixedCoupon, floatingEndCnt, fixedEndCnt, exerciseEndCnt,
                      floatingCnt, fixedCnt, exerciseCnt);

    lattice.rollback(model, engine, endCnt, time, dtime, NPV);
}

/**
 * @brief Tree Swaption Pricing Engine using Trinomial Tree based 2D Lattice method.
 *
 * @tparam DT supported data type including double and float data type, which decides the precision of result.
 * @tparam Model short-rate model class
 * @tparam Process stochastic process class
 * @tparam DIM 1D or 2D short-rate model
 * @tparam LEN maximum length of timestep, which affects the latency and resources utilization.
 * @tparam LEN2 maximum length of node of tree, which affects the latency and resources utilization.
 *
 * @param model short-rate model that has been initialized
 * @param process1 1st dimensional parameters of stochastic process
 * @param process2 2nd dimensional parameters of stochastic process
 * @param type 0: Payer, 1: Receiver
 * @param fixedRate fixed annual interest rate.
 * @param timestep estimate the number of discrete steps from 0 to T, T is the expiry time.
 * @param initTime the time including begin timepoint, end timepoint, exercise timepoints, floating coupon timepoints,
 * and fixed coupon timepoints is arranged from small to large. The timepoints are relative values based on the
 * reference date the unit is year.
 * @param initSize the length of array initTime.
 * @param exerciseCnt exercise timepoints count in initTime.
 * @param floatingCnt floating coupon timepoints count in initTime.
 * @param fixedCnt fixed coupon timepoints count in initTime.
 * @param flatRate floating benchmark annual interest rate
 * @param nominal nominal principal
 * @param x0 initial underlying
 * @param spread spreads on interest rates
 * @param rho the correlation coefficient between price and variance.
 * @param NPV is pricing result array of this engine
 */
template <typename DT, typename Model, typename Process, int DIM, int LEN, int LEN2>
void treeSwaptionEngine(Model& model,
                        DT* process1,
                        DT* process2,
                        int type,
                        DT fixedRate,
                        int timestep,
                        DT initTime[LEN],
                        int initSize,
                        int* exerciseCnt,
                        int* floatingCnt,
                        int* fixedCnt,
                        DT flatRate,
                        DT nominal,
                        DT x0,
                        DT spread,
                        DT rho,
                        DT* NPV) {
    DT time[LEN];
    DT dtime[LEN];
    int beginCnt;
    int endCnt;
    int exerciseEndCnt;
    int floatingEndCnt;
    int fixedEndCnt;

    internal::TimeGrid<DT, LEN> grid;

    DT dtMax = initTime[initSize - 1] / timestep;
    grid.calcuGrid(initSize, initTime, dtMax, time, dtime, exerciseEndCnt, exerciseCnt, fixedEndCnt, floatingEndCnt,
                   fixedCnt, floatingCnt, endCnt);

#ifndef __SYNTHESIS__
    cout << "set timesteps=" << timestep << ",actual timesteps=" << endCnt + 1 << endl;
#endif

    // define TreeLattice as lattice
    DT fixedCoupon = nominal * fixedRate;
    DT accruedSpread = 0.0; // nominal * T * spread;

    xf::fintech::TreeLattice<DT, Model, Process, internal::TreeInstrument<DT, 0, LEN2>, DIM, LEN, LEN2> lattice;
    lattice.setup(model, process1, process2, endCnt + 1, flatRate, x0, time, dtime);

    internal::TreeInstrument<DT, 0, LEN2> engine;

#ifndef __SYNTHESIS__
    DT corr = std::abs(rho);
#else
    DT corr = hls::abs(rho);
#endif

    engine.initialize(type, corr, nominal, accruedSpread, fixedCoupon, floatingEndCnt, fixedEndCnt, exerciseEndCnt,
                      floatingCnt, fixedCnt, exerciseCnt);

    lattice.rollback(model, engine, endCnt, time, dtime, NPV);
}

/**
 * @brief Tree Swap Pricing Engine using Trinomial Tree based 1D Lattice method.
 *
 * @tparam DT supported data type including double and float data type, which decides the precision of result.
 * @tparam Model short-rate model
 * @tparam Process stochastic process
 * @tparam DIM 1D or 2D short-rate model
 * @tparam LEN maximum length of timestep, which affects the latency and resources utilization.
 * @tparam LEN2 maximum length of node of tree, which affects the latency and resources utilization.
 *
 * @param model short-rate model that has been initialized
 * @param process parameters of stochastic process
 * @param type 0: Payer, 1: Receiver
 * @param fixedRate fixed annual interest rate.
 * @param timestep estimate the number of discrete steps from 0 to T, T is the expiry time.
 * @param initTime the time including begin timepoint, end timepoint, exercise timepoints, floating coupon timepoints,
 * and fixed coupon timepoints is arranged from small to large. The timepoints are relative values based on the
 * reference date the unit is year.
 * @param initSize the length of array initTime.
 * @param floatingCnt floating coupon timepoints count in initTime.
 * @param fixedCnt fixed coupon timepoints count in initTime.
 * @param flatRate floating benchmark annual interest rate
 * @param nominal nominal principal
 * @param x0 initial underlying
 * @param spread spreads on interest rates
 * @param NPV is pricing result array of this engine
 */
template <typename DT, typename Model, typename Process, int DIM, int LEN, int LEN2>
void treeSwapEngine(Model& model,
                    DT* process,
                    int type,
                    DT fixedRate,
                    int timestep,
                    DT initTime[LEN],
                    int initSize,
                    int* floatingCnt,
                    int* fixedCnt,
                    DT flatRate,
                    DT nominal,
                    DT x0,
                    DT spread,
                    DT* NPV) {
    DT time[LEN];
    DT dtime[LEN];
    int beginCnt;
    int endCnt;
    int floatingEndCnt;
    int fixedEndCnt;

    internal::TimeGrid<DT, LEN> grid;

    DT dtMax = initTime[initSize - 1] / timestep;
    grid.calcuGrid(initSize, initTime, dtMax, time, dtime, fixedEndCnt, floatingEndCnt, fixedCnt, floatingCnt, endCnt);

#ifndef __SYNTHESIS__
    cout << "set timesteps=" << timestep << ",actual timesteps=" << endCnt + 1 << endl;
#endif

    typedef internal::TreeInstrument<DT, 1, LEN2> Engine;
    // define TreeLattice as lattice
    DT fixedCoupon = nominal * fixedRate;
    DT accruedSpread = 0.0; // nominal * T * spread;

    xf::fintech::TreeLattice<DT, Model, Process, Engine, 1, LEN, LEN2> lattice;
    lattice.setup(model, process, endCnt + 1, flatRate, x0, time, dtime);

    Engine engine;
    engine.initialize(type, nominal, accruedSpread, fixedCoupon, floatingEndCnt, fixedEndCnt, floatingCnt, fixedCnt);

    lattice.rollback(model, engine, endCnt, time, dtime, NPV);
}

/**
 * @brief Tree Swap Pricing Engine using Trinomial Tree based 2D Lattice method.
 *
 * @tparam DT supported data type including double and float data type, which decides the precision of result.
 * @tparam Model short-rate model
 * @tparam Process stochastic process
 * @tparam DIM 1D or 2D short-rate model
 * @tparam LEN maximum length of timestep, which affects the latency and resources utilization.
 * @tparam LEN2 maximum length of node of tree, which affects the latency and resources utilization.
 *
 * @param model short-rate model that has been initialized
 * @param process1 1st dimensional parameters of stochastic process
 * @param process2 2nd dimensional parameters of stochastic process
 * @param type 0: Payer, 1: Receiver
 * @param fixedRate fixed annual interest rate.
 * @param timestep estimate the number of discrete steps from 0 to T, T is the expiry time.
 * @param initTime the time including begin timepoint, end timepoint, exercise timepoints, floating coupon timepoints,
 * and fixed coupon timepoints is arranged from small to large. The timepoints are relative values based on the
 * reference date the unit is year.
 * @param initSize the length of array initTime.
 * @param floatingCnt floating coupon timepoints count in initTime.
 * @param fixedCnt fixed coupon timepoints count in initTime.
 * @param flatRate floating benchmark annual interest rate
 * @param nominal nominal principal
 * @param x0 initial underlying
 * @param spread spreads on interest rates
 * @param rho the correlation coefficient between price and variance.
 * @param NPV is pricing result array of this engine
 */
template <typename DT, typename Model, typename Process, int DIM, int LEN, int LEN2>
void treeSwapEngine(Model& model,
                    DT* process1,
                    DT* process2,
                    int type,
                    DT fixedRate,
                    int timestep,
                    DT initTime[LEN],
                    int initSize,
                    int* floatingCnt,
                    int* fixedCnt,
                    DT flatRate,
                    DT nominal,
                    DT x0,
                    DT spread,
                    DT rho,
                    DT* NPV) {
    DT time[LEN];
    DT dtime[LEN];
    int beginCnt;
    int endCnt;
    int floatingEndCnt;
    int fixedEndCnt;

    internal::TimeGrid<DT, LEN> grid;

    DT dtMax = initTime[initSize - 1] / timestep;
    grid.calcuGrid(initSize, initTime, dtMax, time, dtime, fixedEndCnt, floatingEndCnt, fixedCnt, floatingCnt, endCnt);

#ifndef __SYNTHESIS__
    cout << "set timesteps=" << timestep << ",actual timesteps=" << endCnt + 1 << endl;
#endif

    typedef internal::TreeInstrument<DT, 1, LEN2> Engine;
    // define TreeLattice as lattice
    DT fixedCoupon = nominal * fixedRate;
    DT accruedSpread = 0.0; // nominal * T * spread;

    xf::fintech::TreeLattice<DT, Model, Process, Engine, DIM, LEN, LEN2> lattice;
    lattice.setup(model, process1, process2, endCnt + 1, flatRate, x0, time, dtime);

    Engine engine;
    engine.initialize(type, rho, nominal, accruedSpread, fixedCoupon, floatingEndCnt, fixedEndCnt, floatingCnt,
                      fixedCnt);

    lattice.rollback(model, engine, endCnt, time, dtime, NPV);
}

/**
 * @brief Tree CapFloor Pricing Engine using Trinomial Tree based 1D Lattice method.
 *
 * @tparam DT supported data type including double and float data type, which decides the precision of result.
 * @tparam Model short-rate model that has been initialized
 * @tparam Process parameters of stochastic process
 * @tparam DIM 1D or 2D short-rate model
 * @tparam LEN maximum length of timestep, which affects the latency and resources utilization.
 * @tparam LEN2 maximum length of node of tree, which affects the latency and resources utilization.
 *
 * @param model short-rate model
 * @param process stochastic process
 * @param type 0: Cap, 1: Collar, 2: Floor
 * @param fixedRate fixed annual interest rate.
 * @param timestep estimate the number of discrete steps from 0 to T, T is the expiry time.
 * @param initTime the time including begin timepoint, end timepoint, exercise timepoints, floating coupon timepoints,
 * and fixed coupon timepoints is arranged from small to large. The timepoints are relative values based on the
 * reference date the unit is year.
 * @param initSize the length of array initTime.
 * @param floatingCnt floating coupon timepoints count in initTime.
 * @param flatRate floating benchmark annual interest rate
 * @param nominal nominal principal
 * @param cfRate cap rate and floor rate
 * @param x0 initial underlying
 * @param spread spreads on interest rates
 * @param NPV is pricing result array of this engine
 */
template <typename DT, typename Model, typename Process, int DIM, int LEN, int LEN2>
void treeCapFloorEngine(Model& model,
                        DT* process,
                        int type,
                        DT fixedRate,
                        int timestep,
                        DT initTime[LEN],
                        int initSize,
                        int* floatingCnt,
                        DT flatRate,
                        DT nominal,
                        DT* cfRate,
                        DT x0,
                        DT spread,
                        DT* NPV) {
    DT time[LEN];
    DT dtime[LEN];
    int beginCnt;
    int endCnt;
    int floatingEndCnt;
    int fixedEndCnt;

    internal::TimeGrid<DT, LEN> grid;

    DT dtMax = initTime[initSize - 1] / timestep;
    grid.calcuGrid(initSize, initTime, dtMax, time, dtime, floatingEndCnt, floatingCnt, endCnt);

#ifndef __SYNTHESIS__
    cout << "set timesteps=" << timestep << ",actual timesteps=" << endCnt + 1 << endl;
#endif

    typedef internal::TreeInstrument<DT, 2, LEN2> Engine;
    // define TreeLattice as lattice
    DT fixedCoupon = nominal * fixedRate;
    DT accruedSpread = 0.0; // nominal * T * spread;

    xf::fintech::TreeLattice<DT, Model, Process, Engine, DIM, LEN, LEN2> lattice;
    lattice.setup(model, process, endCnt + 1, flatRate, x0, time, dtime);

    Engine engine;
    engine.initialize(type, initTime[initSize - 1], nominal, cfRate, floatingEndCnt, floatingCnt);

    lattice.rollback(model, engine, endCnt, time, dtime, NPV);
}

/**
 * @brief Tree CapFloor Pricing Engine using Trinomial Tree based 2D Lattice method.
 *
 * @tparam DT supported data type including double and float data type, which decides the precision of result.
 * @tparam Model short-rate model
 * @tparam Process stochastic process
 * @tparam DIM 1D or 2D short-rate model
 * @tparam LEN maximum length of timestep, which affects the latency and resources utilization.
 * @tparam LEN2 maximum length of node of tree, which affects the latency and resources utilization.
 *
 * @param model short-rate model that has been initialized
 * @param process1 1st dimensional parameters of stochastic process
 * @param process2 2nd dimensional parameters of stochastic process
 * @param type 0: Cap, 1: Collar, 2: Floor
 * @param fixedRate fixed annual interest rate.
 * @param timestep estimate the number of discrete steps from 0 to T, T is the expiry time.
 * @param initTime the time including begin timepoint, end timepoint, exercise timepoints, floating coupon timepoints,
 * and fixed coupon timepoints is arranged from small to large. The timepoints are relative values based on the
 * reference date the unit is year.
 * @param initSize the length of array initTime.
 * @param floatingCnt floating coupon timepoints count in initTime.
 * @param flatRate floating benchmark annual interest rate
 * @param nominal nominal principal
 * @param cfRate cap rate ans floor rate
 * @param x0 initial underlying
 * @param spread spreads on interest rates
 * @param rho the correlation coefficient between price and variance.
 * @param NPV is pricing result array of this engine
 */
template <typename DT, typename Model, typename Process, int DIM, int LEN, int LEN2>
void treeCapFloorEngine(Model& model,
                        DT* process1,
                        DT* process2,
                        int type,
                        DT fixedRate,
                        int timestep,
                        DT initTime[LEN],
                        int initSize,
                        int* floatingCnt,
                        DT flatRate,
                        DT nominal,
                        DT* cfRate,
                        DT x0,
                        DT spread,
                        DT rho,
                        DT* NPV) {
    DT time[LEN];
    DT dtime[LEN];
    int beginCnt;
    int endCnt;
    int floatingEndCnt;
    int fixedEndCnt;

    internal::TimeGrid<DT, LEN> grid;

    DT dtMax = initTime[initSize - 1] / timestep;
    grid.calcuGrid(initSize, initTime, dtMax, time, dtime, floatingEndCnt, floatingCnt, endCnt);

#ifndef __SYNTHESIS__
    cout << "set timesteps=" << timestep << ",actual timesteps=" << endCnt + 1 << endl;
#endif

    typedef internal::TreeInstrument<DT, 2, LEN2> Engine;
    // define TreeLattice as lattice
    DT fixedCoupon = nominal * fixedRate;
    DT accruedSpread = 0.0; // nominal * T * spread;

    xf::fintech::TreeLattice<DT, Model, Process, Engine, DIM, LEN, LEN2> lattice;
    lattice.setup(model, process1, process2, endCnt + 1, flatRate, x0, time, dtime);

    Engine engine;
    engine.initialize(type, rho, initTime[initSize - 1], nominal, cfRate, floatingEndCnt, floatingCnt);

    lattice.rollback(model, engine, endCnt, time, dtime, NPV);
}

/**
 * @brief Tree Callable Fixed Rate Bond Pricing Engine using Trinomial Tree based 1D Lattice method.
 *
 * @tparam DT supported data type including double and float data type, which decides the precision of result.
 * @tparam Model short-rate model
 * @tparam Process stochastic process
 * @tparam DIM 1D or 2D short-rate model
 * @tparam LEN maximum length of timestep, which affects the latency and resources utilization.
 * @tparam LEN2 maximum length of node of tree, which affects the latency and resources utilization.
 *
 * @param model short-rate model that has been initialized
 * @param process parameters of stochastic process
 * @param type type of the callability, 0: Call, 1: Put
 * @param fixedRate fixed annual interest rate.
 * @param timestep estimate the number of discrete steps from 0 to T, T is the expiry time.
 * @param initTime the time including begin timepoint, end timepoint, exercise timepoints, floating coupon timepoints,
 * and fixed coupon timepoints is arranged from small to large. The timepoints are relative values based on the
 * reference date the unit is year.
 * @param initSize the length of array initTime.
 * @param callableCnt callable timepoints count in initTime.
 * @param paymentCnt payment timepoints count in initTime.
 * @param flatRate floating benchmark annual interest rate
 * @param nominal nominal principal
 * @param x0 initial underlying
 * @param spread spreads on interest rates
 * @param NPV is pricing result array of this engine
 */
template <typename DT, typename Model, typename Process, int DIM, int LEN, int LEN2>
void treeCallableEngine(Model& model,
                        DT* process,
                        int type,
                        DT fixedRate,
                        int timestep,
                        DT initTime[LEN],
                        int initSize,
                        int* callableCnt,
                        int* paymentCnt,
                        DT flatRate,
                        DT nominal,
                        DT x0,
                        DT spread,
                        DT* NPV) {
    DT time[LEN];
    DT dtime[LEN];
    int beginCnt;
    int endCnt;
    int callableEndCnt;
    int paymentEndCnt;

    internal::TimeGrid<DT, LEN> grid;

    DT dtMax = initTime[initSize - 1] / timestep;
    grid.calcuGrid(initSize, initTime, dtMax, time, dtime, paymentEndCnt, callableEndCnt, paymentCnt, callableCnt,
                   endCnt);

#ifndef __SYNTHESIS__
    cout << "set timesteps=" << timestep << ",actual timesteps=" << endCnt + 1 << endl;
#endif

    typedef internal::TreeInstrument<DT, 3, LEN2> Engine;
    // define TreeLattice as lattice
    DT fixedCoupon = nominal * fixedRate;

    xf::fintech::TreeLattice<DT, Model, Process, Engine, DIM, LEN, LEN2> lattice;
    lattice.setup(model, process, endCnt + 1, flatRate, x0, time, dtime);

    Engine engine;
    engine.initialize(type, nominal, fixedCoupon, callableEndCnt, paymentEndCnt, callableCnt, paymentCnt);

    lattice.rollback(model, engine, endCnt, time, dtime, NPV);
}

/**
 * @brief Tree Callable Fixed Rate Bond Pricing Engine using Trinomial Tree based 2D Lattice method.
 *
 * @tparam DT supported data type including double and float data type, which decides the precision of result.
 * @tparam Model short-rate model
 * @tparam Process stochastic process
 * @tparam DIM 1D or 2D short-rate model
 * @tparam LEN maximum length of timestep, which affects the latency and resources utilization.
 * @tparam LEN2 maximum length of node of tree, which affects the latency and resources utilization.
 *
 * @param model short-rate model that has been initialized
 * @param process1 1st dimensional parameters of stochastic process
 * @param process2 2nd dimensional parameters of stochastic process
 * @param type type of the callability, 0: Call, 1: Put
 * @param fixedRate fixed annual interest rate.
 * @param timestep estimate the number of discrete steps from 0 to T, T is the expiry time.
 * @param initTime the time including begin timepoint, end timepoint, exercise timepoints, floating coupon timepoints,
 * and fixed coupon timepoints is arranged from small to large. The timepoints are relative values based on the
 * reference date the unit is year.
 * @param initSize the length of array initTime.
 * @param callableCnt callable timepoints count in initTime.
 * @param paymentCnt payment timepoints count in initTime.
 * @param flatRate floating benchmark annual interest rate
 * @param nominal nominal principal
 * @param x0 initial underlying
 * @param spread spreads on interest rates
 * @param rho the correlation coefficient between price and variance.
 * @param NPV is pricing result array of this engine
 */
template <typename DT, typename Model, typename Process, int DIM, int LEN, int LEN2>
void treeCallableEngine(Model& model,
                        DT* process1,
                        DT* process2,
                        int type,
                        DT fixedRate,
                        int timestep,
                        DT initTime[LEN],
                        int initSize,
                        int* callableCnt,
                        int* paymentCnt,
                        DT flatRate,
                        DT nominal,
                        DT x0,
                        DT spread,
                        DT rho,
                        DT* NPV) {
    DT time[LEN];
    DT dtime[LEN];
    int beginCnt;
    int endCnt;
    int callableEndCnt;
    int paymentEndCnt;

    internal::TimeGrid<DT, LEN> grid;

    DT dtMax = initTime[initSize - 1] / timestep;
    grid.calcuGrid(initSize, initTime, dtMax, time, dtime, paymentEndCnt, callableEndCnt, paymentCnt, callableCnt,
                   endCnt);

#ifndef __SYNTHESIS__
    cout << "set timesteps=" << timestep << ",actual timesteps=" << endCnt + 1 << endl;
#endif

    typedef internal::TreeInstrument<DT, 3, LEN2> Engine;
    // define TreeLattice as lattice
    DT fixedCoupon = nominal * fixedRate;

    xf::fintech::TreeLattice<DT, Model, Process, Engine, DIM, LEN, LEN2> lattice;
    lattice.setup(model, process1, process2, endCnt + 1, flatRate, x0, time, dtime);

    Engine engine;
    engine.initialize(type, rho, nominal, fixedCoupon, callableEndCnt, paymentEndCnt, callableCnt, paymentCnt);

    lattice.rollback(model, engine, endCnt, time, dtime, NPV);
}

} // fintech
} // xf

#endif
