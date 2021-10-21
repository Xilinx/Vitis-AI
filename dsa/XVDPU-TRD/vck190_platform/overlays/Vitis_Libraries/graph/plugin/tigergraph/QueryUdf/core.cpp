/*
 * Copyright 2020 Xilinx, Inc.
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

#include "loader.hpp"
#include <dlfcn.h>
#include <iostream>
#include "codevector.hpp"

int bfs_fpga_wrapper(int numVertices,
                     int numEdges,
                     int sourceID,
                     xf::graph::Graph<unsigned int, unsigned int> g,
                     unsigned int* predecent,
                     unsigned int* distance) {
    std::cout << "INFO: Running Breadth-First Search API\n\n";

    // open the library
    std::cout << "INFO: Opening libgraphL3wrapper.so...\n";
    std::string basePath = TIGERGRAPH_PATH;
    std::string SOFILEPATH = basePath + "/dev/gdk/gsql/src/QueryUdf/libgraphL3wrapper.so";
    void* handle = dlopen(SOFILEPATH.c_str(), RTLD_LAZY | RTLD_GLOBAL);

    if (!handle) {
        std::cerr << "ERROR: Cannot open library: " << dlerror() << '\n';
        return 1;
    }

    // load the symbol
    std::cout << "INFO: Loading symbol bfs_fpga...\n";
    typedef void (*runKernel_t)(int, int, int, xf::graph::Graph<unsigned int, unsigned int>, unsigned int*,
                                unsigned int*);

    // reset errors
    dlerror();

    runKernel_t runT = (runKernel_t)dlsym(handle, "bfs_fpga");
    const char* dlsym_error2 = dlerror();
    if (dlsym_error2) {
        std::cerr << "ERROR: Cannot load symbol 'bfs_fpga': " << dlsym_error2 << '\n';
        dlclose(handle);
        return 1;
    }

    // use it to do the calculation
    std::cout << "INFO: Calling 'bfs_fpga'...\n";
    runT(numVertices, numEdges, sourceID, g, predecent, distance);

    // close the library
    std::cout << "INFO: Closing library...\n";
    dlclose(handle);
    return 0;
}

int load_xgraph_fpga_wrapper(uint32_t numVertices, uint32_t numEdges, xf::graph::Graph<uint32_t, float> g) {
    std::cout << "INFO: Running Load Graph of Single Source Shortest Path API\n\n";

    // open the library
    std::cout << "INFO: Opening libgraphL3wrapper.so...\n";
    std::string basePath = TIGERGRAPH_PATH;
    std::string SOFILEPATH = basePath + "/dev/gdk/gsql/src/QueryUdf/libgraphL3wrapper.so";
    void* handle = dlopen(SOFILEPATH.c_str(), RTLD_LAZY | RTLD_GLOBAL);

    if (!handle) {
        std::cerr << "ERROR: Cannot open library: " << dlerror() << '\n';
        return 1;
    }

    // load the symbol
    std::cout << "INFO: Loading symbol load_xgraph_fpga...\n";
    typedef void (*runKernel_t)(int, int, xf::graph::Graph<unsigned int, float>);

    // reset errors
    dlerror();

    runKernel_t runT = (runKernel_t)dlsym(handle, "load_xgraph_fpga");
    const char* dlsym_error2 = dlerror();
    if (dlsym_error2) {
        std::cerr << "ERROR: Cannot load symbol 'load_xgraph_fpga': " << dlsym_error2 << '\n';
        dlclose(handle);
        return 1;
    }

    // use it to do the calculation
    std::cout << "INFO: Calling 'load_xgraph_fpga'...\n";
    runT(numVertices, numEdges, g);

    // close the library
    std::cout << "INFO: Closing library...\n";
    dlclose(handle);
    return 0;
}

int shortest_ss_pos_wt_fpga_wrapper(uint32_t numVertices,
                                    uint32_t sourceID,
                                    bool weighted,
                                    xf::graph::Graph<uint32_t, float> g,
                                    float** result,
                                    uint32_t** pred) {
    std::cout << "INFO: Running Single Source Shortest Path API\n\n";

    // open the library
    std::cout << "INFO: Opening libgraphL3wrapper.so...\n";
    std::string basePath = TIGERGRAPH_PATH;
    std::string SOFILEPATH = basePath + "/dev/gdk/gsql/src/QueryUdf/libgraphL3wrapper.so";
    void* handle = dlopen(SOFILEPATH.c_str(), RTLD_LAZY | RTLD_GLOBAL);

    if (!handle) {
        std::cerr << "ERROR: Cannot open library: " << dlerror() << '\n';
        return 1;
    }

    // load the symbol
    std::cout << "INFO: Loading symbol shortest_ss_pos_wt_fpga...\n";
    typedef void (*runKernel_t)(int, int, bool, xf::graph::Graph<unsigned int, float>, float**, unsigned int**);

    // reset errors
    dlerror();

    runKernel_t runT = (runKernel_t)dlsym(handle, "shortest_ss_pos_wt_fpga");
    const char* dlsym_error2 = dlerror();
    if (dlsym_error2) {
        std::cerr << "ERROR: Cannot load symbol 'shortest_ss_pos_wt_fpga': " << dlsym_error2 << '\n';
        dlclose(handle);
        return 1;
    }

    // use it to do the calculation
    std::cout << "INFO: Calling 'shortest_ss_pos_wt_fpga'...\n";
    runT(numVertices, sourceID, weighted, g, result, pred);

    // close the library
    std::cout << "INFO: Closing library...\n";
    dlclose(handle);
    return 0;
}

int load_xgraph_pageRank_wt_fpga_wrapper(uint32_t numVertices, uint32_t numEdges, xf::graph::Graph<uint32_t, float> g) {
    std::cout << "INFO: Running Load Graph of PageRank API\n\n";

    // open the library
    std::cout << "INFO: Opening libgraphL3wrapper.so...\n";
    std::string basePath = TIGERGRAPH_PATH;
    std::string SOFILEPATH = basePath + "/dev/gdk/gsql/src/QueryUdf/libgraphL3wrapper.so";
    void* handle = dlopen(SOFILEPATH.c_str(), RTLD_LAZY | RTLD_GLOBAL);

    if (!handle) {
        std::cerr << "ERROR: Cannot open library: " << dlerror() << '\n';
        return 1;
    }

    // load the symbol
    std::cout << "INFO: Loading symbol load_xgraph_pageRank_wt_fpga...\n";
    typedef void (*runKernel_t)(int, int, xf::graph::Graph<unsigned int, float>);

    // reset errors
    dlerror();

    runKernel_t runT = (runKernel_t)dlsym(handle, "load_xgraph_pageRank_wt_fpga");
    const char* dlsym_error2 = dlerror();
    if (dlsym_error2) {
        std::cerr << "ERROR: Cannot load symbol 'load_xgraph_pageRank_wt_fpga': " << dlsym_error2 << '\n';
        dlclose(handle);
        return 1;
    }

    // use it to do the calculation
    std::cout << "INFO: Calling 'load_xgraph_pageRank_wt_fpga'...\n";
    runT(numVertices, numEdges, g);

    // close the library
    std::cout << "INFO: Closing library...\n";
    dlclose(handle);
    return 0;
}

int pageRank_wt_fpga_wrapper(
    float alpha, float tolerance, uint32_t maxIter, xf::graph::Graph<uint32_t, float> g, float* rank) {
    std::cout << "INFO: Running PageRank API\n\n";

    // open the library
    std::cout << "INFO: Opening libgraphL3wrapper.so...\n";
    std::string basePath = TIGERGRAPH_PATH;
    std::string SOFILEPATH = basePath + "/dev/gdk/gsql/src/QueryUdf/libgraphL3wrapper.so";
    void* handle = dlopen(SOFILEPATH.c_str(), RTLD_LAZY | RTLD_GLOBAL);

    if (!handle) {
        std::cerr << "ERROR: Cannot open library: " << dlerror() << '\n';
        return 1;
    }

    // load the symbol
    std::cout << "INFO: Loading symbol pageRank_wt_fpga...\n";
    typedef void (*runKernel_t)(float, float, uint32_t, xf::graph::Graph<unsigned int, float>, float*);

    // reset errors
    dlerror();

    runKernel_t runT = (runKernel_t)dlsym(handle, "pageRank_wt_fpga");
    const char* dlsym_error2 = dlerror();
    if (dlsym_error2) {
        std::cerr << "ERROR: Cannot load symbol 'pageRank_wt_fpga': " << dlsym_error2 << '\n';
        dlclose(handle);
        return 1;
    }

    // use it to do the calculation
    std::cout << "INFO: Calling 'pageRank_wt_fpga'...\n";
    runT(alpha, tolerance, maxIter, g, rank);

    // close the library
    std::cout << "INFO: Closing library...\n";
    dlclose(handle);
    return 0;
}

int load_xgraph_cosine_nbor_ss_fpga_wrapper(uint32_t numVertices,
                                            uint32_t numEdges,
                                            xf::graph::Graph<uint32_t, float> g) {
    std::cout << "INFO: Running Load Graph of Single Source Cosine Similarity API\n\n";

    // open the library
    std::cout << "INFO: Opening libgraphL3wrapper.so...\n";
    std::string basePath = TIGERGRAPH_PATH;
    std::string SOFILEPATH = basePath + "/dev/gdk/gsql/src/QueryUdf/libgraphL3wrapper.so";
    void* handle = dlopen(SOFILEPATH.c_str(), RTLD_LAZY | RTLD_GLOBAL);

    if (!handle) {
        std::cerr << "ERROR: Cannot open library: " << dlerror() << '\n';
        return 1;
    }

    // load the symbol
    std::cout << "INFO: Loading symbol load_xgraph_cosine_nbor_ss_fpga...\n";
    typedef void (*runKernel_t)(int, int, xf::graph::Graph<unsigned int, float>);

    // reset errors
    dlerror();

    runKernel_t runT = (runKernel_t)dlsym(handle, "load_xgraph_cosine_nbor_ss_fpga");
    const char* dlsym_error2 = dlerror();
    if (dlsym_error2) {
        std::cerr << "ERROR: Cannot load symbol 'load_xgraph_cosine_nbor_ss_fpga': " << dlsym_error2 << '\n';
        dlclose(handle);
        return 1;
    }

    // use it to do the calculation
    std::cout << "INFO: Calling 'load_xgraph_cosine_nbor_ss_fpga'...\n";
    runT(numVertices, numEdges, g);

    // close the library
    std::cout << "INFO: Closing library...\n";
    dlclose(handle);
    return 0;
}

int cosine_nbor_ss_fpga_wrapper(uint32_t topK,
                                uint32_t sourceLen,
                                uint32_t* sourceIndice,
                                uint32_t* sourceWeight,
                                xf::graph::Graph<uint32_t, float> g,
                                uint32_t* resultID,
                                float* similarity) {
    std::cout << "INFO: Running Single Source Cosine Similarity API\n\n";

    // open the library
    std::cout << "INFO: Opening libgraphL3wrapper.so...\n";
    std::string basePath = TIGERGRAPH_PATH;
    std::string SOFILEPATH = basePath + "/dev/gdk/gsql/src/QueryUdf/libgraphL3wrapper.so";
    void* handle = dlopen(SOFILEPATH.c_str(), RTLD_LAZY | RTLD_GLOBAL);

    if (!handle) {
        std::cerr << "ERROR: Cannot open library: " << dlerror() << '\n';
        return 1;
    }

    // load the symbol
    std::cout << "INFO: Loading symbol cosine_nbor_ss_fpga...\n";
    typedef void (*runKernel_t)(uint32_t, uint32_t, uint32_t*, uint32_t*, xf::graph::Graph<unsigned int, float>,
                                uint32_t*, float*);

    // reset errors
    dlerror();

    runKernel_t runT = (runKernel_t)dlsym(handle, "cosine_nbor_ss_fpga");
    const char* dlsym_error2 = dlerror();
    if (dlsym_error2) {
        std::cerr << "ERROR: Cannot load symbol 'cosine_nbor_ss_fpga': " << dlsym_error2 << '\n';
        dlclose(handle);
        return 1;
    }

    // use it to do the calculation
    std::cout << "INFO: Calling 'cosine_nbor_ss_fpga'...\n";
    runT(topK, sourceLen, sourceIndice, sourceWeight, g, resultID, similarity);

    // close the library
    std::cout << "INFO: Closing library...\n";
    dlclose(handle);
    return 0;
}

namespace xai {

const unsigned int startPropertyIndex = 3; // Start index of property in the
                                           // vector, 0, 1 and 2 are ressrved
                                           // for norm, id */
CodeToIdMap* pMap = nullptr;
std::vector<uint64_t> IDMap;
std::int64_t abs64(std::int64_t x) {
    return (x >= 0) ? x : -x;
}

/**
* Converts a set of codes into a vector of integers to use in a cosine
* similarity computation
*
* @param concept the SNOMED concept (an arbitrary unsigned integer) to which the
* codes belong
* @param vectorLength the number of elements to generate for the cosine vector
* @param codes the set of codes to encode into the output vector.  The vector
* does not need to be sorted
* @return a vector of integers to embed into the cosine similarity vector
*
* See
* https://confluence.xilinx.com/display/DBA/Mapping+SNOMED+CT+Codes+into+Vector
* for details.
*/

std::vector<CosineVecValue> makeCosineVector(SnomedConcept concept,
                                             unsigned vectorLength,
                                             const std::vector<SnomedCode>& codes) {
    std::vector<CosineVecValue> outVec;
    outVec.reserve(vectorLength);
    CodeToIdMap* pIdMap = CodeToIdMap::getInstance();
    // const SnomedId numIds = pIdMap->getNumReservedIds(concept);
    SnomedId numIds = pIdMap->getNumReservedIds(concept);
    if (numIds < vectorLength) numIds = vectorLength;

    // Create the set of buckets

    std::vector<Bucket> buckets;
    for (std::uint64_t i = 0; i < vectorLength; ++i) {
        SnomedId pos = SnomedId(i * numIds / vectorLength);
        SnomedId nextPos = SnomedId((i + 1) * numIds / vectorLength);
        SnomedId bucketStartId = pos;
        SnomedId bucketEndId = (nextPos >= numIds) ? numIds : nextPos;
        buckets.emplace_back(bucketStartId, bucketEndId, MinVecValue, MaxVecValue, NullVecValue);
    }
    // Fill the buckets with codes.  When a bucket is full, dump the summary of
    // the bucket to the vector.

    for (unsigned i = 0; i < codes.size(); ++i) {
        // Convert the code to an ID (small int)
        SnomedId id = pIdMap->addCode(concept, codes[i]);

        // Determine which bucket the ID goes into.  If the ID is out of range
        // (because it wasn't accounted for when
        // IDs were reserved for the concept), ignore the ID/code.
        const unsigned bucketNum = id * vectorLength / numIds;
        if (bucketNum >= vectorLength) continue;

        // Add the ID to the bucket
        buckets[bucketNum].addId(id);
    }
    // Dump all buckets and return the vector

    for (const Bucket& bucket : buckets) outVec.push_back(bucket.getCosineVecVale());

    return outVec; // Move semantics
}
}

int loadgraph_cosinesim_ss_dense_fpga_wrapper(uint32_t deviceNeeded,
                                              uint32_t cuNm,
                                              xf::graph::Graph<int32_t, int32_t>** g) {
    int status = 0;
    std::cout << "INFO: Running Load Graph for Single Source Cosine Similarity Dense API\n\n";

    // open the library
    std::cout << "INFO: Opening libgraphL3wrapper.so...\n";
    std::string basePath = TIGERGRAPH_PATH;
    std::string SOFILEPATH = basePath + "/dev/gdk/gsql/src/QueryUdf/libgraphL3wrapper.so";
    void* handle = dlopen(SOFILEPATH.c_str(), RTLD_LAZY | RTLD_GLOBAL);

    if (!handle) {
        std::cerr << "ERROR: Cannot open library: " << dlerror() << '\n';
        return -5;
    }

    // load the symbol
    std::cout << "INFO: Loading symbol loadgraph_cosinesim_ss_dense_fpga...\n";
    typedef int (*load_t)(uint32_t, uint32_t, xf::graph::Graph<int32_t, int32_t>**);

    // reset errors
    dlerror();

    load_t runT = (load_t)dlsym(handle, "loadgraph_cosinesim_ss_dense_fpga");
    const char* dlsym_error2 = dlerror();
    if (dlsym_error2) {
        std::cerr << "ERROR: Cannot load symbol 'loadgraph_cosinesim_ss_dense_fpga': " << dlsym_error2 << '\n';
        dlclose(handle);
        return -6;
    }

    // use it to do the calculation
    std::cout << "INFO: Calling 'loadgraph_cosinesim_ss_dense_fpga'...\n";
    status = runT(deviceNeeded, cuNm, g);

    // close the library
    std::cout << "INFO: Closing library... status=" << status << std::endl;
    dlclose(handle);
    return status;
}

int cosinesim_ss_dense_fpga(uint32_t deviceNeeded,
                            int32_t sourceLen,
                            int32_t* sourceWeight,
                            int32_t topK,
                            xf::graph::Graph<int32_t, int32_t>** g,
                            int32_t* resultID,
                            float* similarity) {
    std::cout << "INFO: Running Single Source Cosine Similarity Dense API\n\n";

    // open the library
    std::cout << "INFO: Opening libgraphL3wrapper.so...\n";
    std::string basePath = TIGERGRAPH_PATH;
    std::string SOFILEPATH = basePath + "/dev/gdk/gsql/src/QueryUdf/libgraphL3wrapper.so";
    void* handle = dlopen(SOFILEPATH.c_str(), RTLD_LAZY | RTLD_GLOBAL);

    if (!handle) {
        std::cerr << "ERROR: Cannot open library: " << dlerror() << '\n';
        return 1;
    }

    // load the symbol
    std::cout << "INFO: Loading symbol cosinesim_ss_dense_fpga...\n";
    typedef void (*runKernel_t)(uint32_t, int32_t, int32_t*, int32_t, xf::graph::Graph<int32_t, int32_t>**, int32_t*,
                                float*);

    // reset errors
    dlerror();

    runKernel_t runT = (runKernel_t)dlsym(handle, "cosinesim_ss_dense_fpga");
    const char* dlsym_error2 = dlerror();
    if (dlsym_error2) {
        std::cerr << "ERROR: Cannot load symbol 'cosinesim_ss_dense_fpga': " << dlsym_error2 << '\n';
        dlclose(handle);
        return 1;
    }

    // use it to do the calculation
    std::cout << "INFO: Calling 'cosinesim_ss_dense_fpga'...\n";
    runT(deviceNeeded, sourceLen, sourceWeight, topK, g, resultID, similarity);

    // close the library
    std::cout << "INFO: Closing library...\n";
    dlclose(handle);
    return 0;
}

int close_fpga() {
    std::cout << "INFO: Closing FPGA\n\n";

    // open the library
    std::cout << "INFO: Opening libgraphL3wrapper.so...\n";
    std::string basePath = TIGERGRAPH_PATH;
    std::string SOFILEPATH = basePath + "/dev/gdk/gsql/src/QueryUdf/libgraphL3wrapper.so";
    void* handle = dlopen(SOFILEPATH.c_str(), RTLD_LAZY | RTLD_GLOBAL);

    if (!handle) {
        std::cerr << "ERROR: Cannot open library: " << dlerror() << '\n';
        return 1;
    }

    // load the symbol
    std::cout << "INFO: Loading symbol close_fpga...\n";
    typedef void (*close_t)();

    // reset errors
    dlerror();

    close_t closeT = (close_t)dlsym(handle, "close_fpga");
    const char* dlsym_error2 = dlerror();
    if (dlsym_error2) {
        std::cerr << "ERROR: Cannot load symbol 'close_fpga': " << dlsym_error2 << '\n';
        dlclose(handle);
        return 1;
    }

    // use it to do the calculation
    std::cout << "INFO: Calling 'close_fpga'...\n";
    closeT();

    // close the library
    std::cout << "INFO: Closing library...\n";
    dlclose(handle);
    return 0;
}
