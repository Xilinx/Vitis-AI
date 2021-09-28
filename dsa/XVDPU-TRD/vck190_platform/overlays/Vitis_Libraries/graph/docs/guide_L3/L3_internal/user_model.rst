.. 
   Copyright 2020 Xilinx, Inc.
  
   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at
  
       http://www.apache.org/licenses/LICENSE-2.0
  
   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.

.. meta::
   :keywords: graph, running flow, asynchronous
   :description: A series of classes are provided that represent the various graph models that are supported. These classes provide all the methods that are required to run that graph model on a given HW device.
   :xlnxdocumentclass: Document
   :xlnxdocumenttype: Tutorials

****************
Running Examples
****************

A series of classes are provided that represent the various graph models that are supported.
These classes provide all the methods that are required to run that graph model on a given HW device.

Basic Flow
**********

The hardware deployments and API executions have been seperated. All the harware deployment related information should be defined in the struct of xf::graph::L3::Handle::singleOP. 
To run an example there are 8 basic steps:

* Instantiate an instance of struct xf::graph::L3::Handle::singleOP
* Instantiate an instance of class xf::graph::L3::Handle
* Deploy hardwares by using the instance of struct singleOP
* Instantiate an instance of class xf::graph::L3::Graph and load data
* Load Graph instance to hardwares
* Run the model (as many times as required)
* Release the Handle instance in order to recycle harware resources
* Release the Graph instance and buffers in order to avoid memory leaks


Example
*******
.. code-block:: c++

    #include "xf_graph_L3.hpp"
    #include <cstring>

    #define DT float

    //----------------- Setup shortestPathFloat thread -------------
    std::string opName;
    std::string kernelName;
    int requestLoad;
    std::string xclbinPath;
    int deviceNeeded;

    xf::graph::L3::Handle::singleOP op0;
    op0.operationName = (char*)opName.c_str();
    op0.setKernelName((char*)kernelName.c_str());
    op0.requestLoad = requestLoad;
    op0.xclbinFile = (char*)xclbinPath.c_str();
    op0.deviceNeeded = deviceNeeded;

    xf::graph::L3::Handle handle0;
    handle0.addOp(op0);
    handle0.setUp();

    //----------------- Readin Graph data ---------------------------
    uint32_t numVertices;
    uint32_t numEdges;
    xf::graph::Graph<uint32_t, DT> g("CSR", numVertices, numEdges);

    //----------------- Load Graph ----------------------------------
    (handle0.opsp)->loadGraph(g);

    //---------------- Run L3 API -----------------------------------
    bool weighted = 1;
    uint32_t nSource = 1;
    uint32_t sourceID = 10;
    uint32_t length = ((numVertices + 1023) / 1024) * 1024;
    DT** result;
    uint32_t** pred;
    result = new DT*[nSource];
    pred = new uint32_t*[nSource];
    for (int i = 0; i < nSource; ++i) {
        result[i] = xf::graph::L3::aligned_alloc<DT>(length);
        pred[i] = xf::graph::L3::aligned_alloc<uint32_t>(length);
    }
    auto ev = xf::graph::L3::shortestPath(handle0, nSource, &sourceID, weighted, g, result, pred);
    int ret = ev.wait();

    //--------------- Free and delete -----------------------------------
    (handle0.opsp)->join();
    handle0.free();
    g.freeBuffers();

    for (int i = 0; i < nSource; ++i) {
        free(result[i]);
        free(pred[i]);
    }
    delete[] result;
    delete[] pred;
  

Asynchronous Execution
**********************

Each API in Graph L3 is implemented in asynchronous mode. So they can receive multi-requests at the same time and if hardware resources are sufficient, different Graph L3 APIs can be executed at the same time. The L3 framework can fully use the hardware resources and achieve high throughput scheduling.   


Example of using multiple requests
----------------------------------
.. code-block:: c++

    #include "xf_graph_L3.hpp"
    #include <cstring>

    #define DT float

    //----------------- Setup shortestPathFloat thread -------------
    std::string opName;
    std::string kernelName;
    int requestLoad;
    std::string xclbinPath;
    int deviceNeeded;

    xf::graph::L3::Handle::singleOP op0;
    op0.operationName = (char*)opName.c_str();
    op0.setKernelName((char*)kernelName.c_str());
    op0.requestLoad = requestLoad;
    op0.xclbinFile = (char*)xclbinPath.c_str();
    op0.deviceNeeded = deviceNeeded;

    //----------------- Setup pageRank thread -------------
    std::string opName2;
    std::string kernelName2;
    int requestLoad2;
    std::string xclbinPath2;
    int deviceNeeded2;

    xf::graph::L3::Handle::singleOP op1;
    op1.operationName = (char*)opName2.c_str();
    op1.setKernelName((char*)kernelName2.c_str());
    op1.requestLoad = requestLoad2;
    op1.xclbinFile = (char*)xclbinPath2.c_str();
    op1.deviceNeeded = deviceNeeded2;

    handle0.addOp(op1);
    handle0.setUp();

    //----------------- Readin Graph data ---------------------------
    uint32_t numVertices;
    uint32_t numEdges;
    xf::graph::Graph<uint32_t, DT> g("CSR", numVertices, numEdges);
    uint32_t numVertices2;
    uint32_t numEdges2;
    xf::graph::Graph<uint32_t, DT> g2("CSR", numVertices2, numEdges2);

    //----------------- Load Graph ----------------------------------
    (handle0.opsp)->loadGraph(g);
    (handle0.oppg)->loadGraph(g2);

    //---------------- Run L3 API -----------------------------------
    auto ev1 = xf::graph::L3::shortestPath(handle0, nSource1, &sourceID1, weighted1, g, result1, pred1);
    auto ev2 = xf::graph::L3::shortestPath(handle0, nSource2, &sourceID2, weighted2, g, result2, pred2);
    auto ev3 = xf::graph::L3::shortestPath(handle0, nSource3, &sourceID3, weighted3, g, result3, pred3);
    auto ev4 = xf::graph::L3::shortestPath(handle0, nSource4, &sourceID5, weighted4, g, result4, pred4);
    auto ev5 = xf::graph::L3::pageRankWeight(handle0, alpha, tolerance, maxIter, g2, pagerank);
    int ret1 = ev1.wait();
    int ret2 = ev2.wait();
    int ret3 = ev3.wait();
    int ret4 = ev4.wait();
    int ret5 = ev5.wait();

    //--------------- Free and delete -----------------------------------
    (handle0.opsp)->join();
    (handle0.oppg)->join();
    handle0.free();
    g.freeBuffers();
    g2.freeBuffers();



