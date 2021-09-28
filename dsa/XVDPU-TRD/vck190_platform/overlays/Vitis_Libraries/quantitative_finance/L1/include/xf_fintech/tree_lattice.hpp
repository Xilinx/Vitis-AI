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
 * @file tree_lattice.hpp
 * @brief This file include the class TreeLattice
 *
 */

#ifndef __XF_FINTECH_LATTICE_HPP_
#define __XF_FINTECH_LATTICE_HPP_

#include "hls_math.h"
#include "ap_int.h"
#include "utils.hpp"
#include "trinomial_tree.hpp"

#ifndef __SYNTHESIS__
#include <iostream>
using namespace std;
#endif

namespace xf {

namespace fintech {

/**
 * @brief Generalized structure compatible with different models and instruments
 *
 * @tparam DT date type supported include float and double.
 * @tparam Model short-rate model for Tree Engine
 * @tparam Process stochastic process
 * @tparam Instrument swaption, swap, cap floor, callable bond
 * @tparam DIM 1D or 2D short-rate model
 * @tparam LEN maximum length of timestep, which affects the latency and resources utilization.
 * @tparam LEN2 maximum length of node of tree, which affects the latency and resources utilization.
 *
 */

template <typename DT, typename Model, typename Process, typename Instrument, int DIM, int LEN, int LEN2>
class TreeLattice {
   public:
    // default constructor
    TreeLattice() {
#pragma HLS inline
    }

    /**
     * @brief setup setup parameter and initialize 1D framework of TreeLattice, and compute short-rate
     *
     * @param model short-rate model for Tree Engine
     * @param process parameters of stochastic process
     * @param endCnt end counter of timepoints
     * @param r floating benchmark annual interest rate
     * @param x0 initial underlying
     * @param time array timepoints
     * @param dtime array the difference between the next timepoint and the current timepoint
     *
     */
    void setup(Model& model, DT* process, unsigned endCnt, DT r, DT x0, DT* time, DT* dtime);

    /**
     * @brief setup setup parameter and initialize 2D framework of TreeLattice, and compute short-rate
     *
     * @param model short-rate model for Tree Engine
     * @param process1 parameters of stochastic process
     * @param process2 parameters of stochastic process
     * @param endCnt end counter of timepoints
     * @param r floating benchmark annual interest rate
     * @param x0 initial underlying
     * @param time array timepoints
     * @param dtime array the difference between the next timepoint and the current timepoint
     *
     */
    void setup(Model& model, DT* process1, DT* process2, unsigned endCnt, DT r, DT x0, DT* time, DT* dtime);

    /**
     * @brief rollback calculate pricing result
     *
     * @param model short-rate model for Tree Engine
     * @param engine swaption, swap, cap floor, callable bond
     * @param from begin counter
     * @param time array timepoints
     * @param dtime array the difference between the next timepoint and the current timepoint
     * @param NPV is pricing result
     *
     */
    void rollback(Model& model, Instrument& engine, unsigned from, DT* time, DT* dtime, DT* NPV);
};

template <typename DT, typename Model, typename Process, typename Instrument, int LEN, int LEN2>
class TreeLattice<DT, Model, Process, Instrument, 1, LEN, LEN2> {
   private:
    // define TrinomialTree as tree
    xf::fintech::TrinomialTree<DT, Process, LEN> tree;
    DT rates[LEN];

    DT tmp_values1[4][LEN2];
    DT tmp_values2[4][LEN2];

   public:
    // default constructor
    TreeLattice() {
#pragma HLS inline
#pragma HLS array_partition variable = tmp_values1 dim = 1 cyclic factor = 4
#pragma HLS array_partition variable = tmp_values2 dim = 1 cyclic factor = 4
#pragma HLS array_partition variable = tmp_values1 dim = 2 cyclic factor = 3
#pragma HLS array_partition variable = tmp_values2 dim = 2 cyclic factor = 2
#pragma HLS resource variable = tmp_values1 core = RAM_2P_BRAM
#pragma HLS resource variable = tmp_values2 core = RAM_2P_BRAM
    }

    // initialize parameter, tree and model
    void setup(Model& model, DT* process, unsigned endCnt, DT r, DT x0, DT* time, DT* dtime) {
#pragma HLS inline
        // trinomial tree initialization
        tree.initialization(process, endCnt, x0);

        // compute short rate
        model.treeShortRate(tree, endCnt, time, dtime, tmp_values1, tmp_values2, tmp_values1[3], rates);
    }

    // calculate present values per nodes
    void rollback(Model& model, Instrument& engine, unsigned from, DT* time, DT* dtime, DT* NPV) {
#pragma HLS inline
        // initialize parameter
        unsigned len_j = 2 * from + 3; // 2(N+1)+1
        engine.reset(len_j, tmp_values1);

        for (int i = from - 1; i >= 0; i--) {
            DT t = time[i];
            DT dt = dtime[i];
            unsigned size = tree.size(i);
            tree.dxUpdateNoCalcu(i, t, dt);
#pragma HLS loop_tripcount min = 50 max = 50
#ifndef __SYNTHESIS__
            cout << "rollback, i=" << i << ",size=" << size << "  ================" << endl;
#endif

            for (unsigned j = 0; j < size; j++) {
#pragma HLS pipeline
#pragma HLS loop_tripcount min = 100 max = 100
                DT probs[3];
                unsigned index = tree.calculateProbability(j, t, dt, probs);
                DT x = tree.underlying(j);
                DT disc = model.discount(t, dt, &x, rates[i]);
                engine.stepback(j, index, disc, probs, tmp_values1, tmp_values2);
            }
            engine.adjustValues(i, size, t, tmp_values1, tmp_values2);
        }
        NPV[0] = tmp_values1[0][0];
    }

}; // class

template <typename DT, typename Model, typename Process, typename Instrument, int LEN, int LEN2>
class TreeLattice<DT, Model, Process, Instrument, 2, LEN, LEN2> {
   private:
    // define TrinomialTree as tree
    xf::fintech::TrinomialTree<DT, Process, LEN> tree[2];
    DT rates[LEN];

    internal::xf_2D_array<DT, 4, LEN2> tmp_values1;
    internal::xf_2D_array<DT, 4, LEN2> tmp_values2;

   public:
    // default constructor
    TreeLattice() {
#pragma HLS inline
#ifdef __SYNTHESIS__
#pragma HLS array_partition variable = tmp_values1._2d_array dim = 1 cyclic factor = 4
#pragma HLS array_partition variable = tmp_values2._2d_array dim = 1 cyclic factor = 4
#pragma HLS array_partition variable = tmp_values1._2d_array dim = 2 cyclic factor = 3
#pragma HLS array_partition variable = tmp_values2._2d_array dim = 2 cyclic factor = 2
#pragma HLS resource variable = tmp_values1 core = XPM_MEMORY uram
#pragma HLS resource variable = tmp_values2 core = XPM_MEMORY uram
#endif
    }

    // initialize parameter, tree and model
    void setup(Model& model, DT* process1, DT* process2, unsigned endCnt, DT r, DT x0, DT* time, DT* dtime) {
#pragma HLS inline

        // trinomial tree initialization
        tree[0].initialization(process1, endCnt, x0);
        tree[1].initialization(process2, endCnt, x0);

        // compute short rate
        model.treeShortRate(tree, endCnt, time, dtime, tmp_values1, tmp_values2, rates);
    }

    // calculate present values per nodes
    void rollback(Model& model, Instrument& engine, int from, DT* time, DT* dtime, DT* NPV) {
#pragma HLS inline
        // initialize parameter
        unsigned len_j = 4 * from * (from + 3) + 9;
        engine.reset(len_j, tmp_values1);
        for (int i = from - 1; i >= 0; i--) {
            DT t = time[i];
            DT dt = dtime[i];
            unsigned size = tree[0].size(i) * tree[1].size(i);
            tree[0].dxUpdateNoCalcu(i, t, dt);
            tree[1].dxUpdateNoCalcu(i, t, dt);
#pragma HLS loop_tripcount min = 50 max = 50
#ifndef __SYNTHESIS__
            cout << "rollback, i=" << i << ",size=" << size << "  ================" << endl;
#endif
            for (unsigned j = 0; j < size; j++) {
#pragma HLS pipeline ii = 2
#pragma HLS loop_tripcount min = 100 max = 100
                DT probs[2][3];
                int modulo = tree[0].size(i);
                int index1 = j % modulo;
                int index2 = j / modulo;
                modulo = tree[0].size(i + 1);
                unsigned k[2];
                k[0] = tree[0].calculateProbability(index1, t, dt, probs[0]);
                k[1] = tree[1].calculateProbability(index2, t, dt, probs[1]);
                DT x[2];
                x[0] = tree[0].underlying(index1);
                x[1] = tree[1].underlying(index2);
                DT disc = model.discount(t, dt, x);
                engine.stepback(j, k, modulo, disc, probs, tmp_values1, tmp_values2);
            }
            engine.adjustValues(i, size, t, tmp_values1, tmp_values2);
        }
        NPV[0] = tmp_values1.read(0, 0);
    }

}; // class

}; // fintech

}; // xf
#endif
