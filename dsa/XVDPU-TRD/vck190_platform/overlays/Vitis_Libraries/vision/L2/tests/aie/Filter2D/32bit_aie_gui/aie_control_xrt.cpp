#include <iostream>
#include "adf/adf_api/AIEControlConfig.h"

/************************** Graph Configurations  *****************************/

adf::GraphConfig GraphConfigurations[] = {
    //{id, name, graphLoadElfFunc, graphInitFunc, graphDebugHalt, coreColumns, coreRows, iterMemColumns, iterMemRows,
    // iterMemAddrs, triggered, plKernelInstanceNames, plAxiLiteModes, plDriverStartFuncs, plDriverCheckIPDoneFuncs}
    {
        0, "filter_graph", nullptr, nullptr, nullptr, {25}, {0}, {24}, {0}, {0x6004}, {0}, {}, {}, {}, {},
    },
};
const int NUM_GRAPH = 1;

/************************** PLIO Configurations  *****************************/

adf::PLIOConfig PLIOConfigurations[] = {
    //{id, name, loginal_name, shim_column, slaveOrMaster, streamId}
    {0, "in1", "DataIn1", 25, 0, 0},
    {1, "out1", "DataOut1", 25, 1, 0},
};
const int NUM_PLIO = 2;

/************************** ADF API initializer *****************************/

class InitializeAIEControlXRT {
   public:
    InitializeAIEControlXRT() {
        std::cout << "Initializing ADF API..." << std::endl;
#ifdef __EXCLUDE_PL_CONTROL__
        bool exclude_pl_control = true;
#else
        bool exclude_pl_control = false;
#endif
        adf::initializeConfigurations(nullptr, 0, 0, 0, GraphConfigurations, NUM_GRAPH, nullptr, 0, nullptr, 0, nullptr,
                                      0, nullptr, 0, nullptr, 0, nullptr, 0, PLIOConfigurations, NUM_PLIO, nullptr, 0,
                                      0, nullptr, false, exclude_pl_control, false, nullptr, true, 2);
    }
} initAIEControlXRT;

#if !defined(__CDO__)

// Kernel Stub Definition
void filter2D(input_window<int>*, output_window<int>*) { /* Stub */
}
#endif
