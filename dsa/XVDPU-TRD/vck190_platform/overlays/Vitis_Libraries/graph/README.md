# Vitis Graph Library

Vitis Graph Library is an open-sourced Vitis library written in C++ for accelerating graph applications in a variety of use cases. It now covers a level of acceleration: the module level (L1), the pre-defined kernel level (L2), the asynchronous software level (L3) and [TigerGraph](https://www.tigergraph.com/) integration (plugin) APIs. 

## Overview

The algorithms implemented by Vitis Graph Library include:

- Similarity analysis: Cosine Similarity, Jaccard Similarity, k-nearest neighbor.
- Centrality analysis: PageRank.
- Pathfinding: Single Source Shortest Path (SSSP), Multi-Sources Shortest Path (MSSP).
- Connectivity analysis: Weakly Connected Components and Strongly Connected Components.
- Community Detection: Louvain Modularity (New for 2021.1), Label Propagation and Triangle Count.
- Search: Breadth First Search and 2-Hop Search (New for 2021.1).
- Graph Format: Calculate Degree and Format Convert between CSR and CSC.

## Benchmark Result

In `L2/benchmarks`, these kernels are built into xclbins targeting Alveo U250/U50. We achieved a good performance on several dataset. For more details about the benchmarks, please find them in [benchmark results](https://xilinx.github.io/Vitis_Libraries/graph/2021.1/benchmark.html).

## Software level API

`L3` offers asynchronous software level APIs. The L3 framework can fully use the hardware resources and achieve high throughput scheduling. And users can send multiple requests at the same time. For details on running these cases, please refer to [Vitis Graph Library Documentation](https://xilinx.github.io/Vitis_Libraries/graph/2021.1/guide_L3/L3_internal/getting_started.html).

## Xilinx TigerGraph Integration

Xilinx TigerGraph Integration is offered in `plugin`. For details on running these cases, please refer to [Vitis Graph Library Documentation](https://xilinx.github.io/Vitis_Libraries/graph/2021.1/plugin/tigergraph_integration.html).

## Documentations

For more details of the Graph library, please refer to [Vitis Graph Library Documentation](https://xilinx.github.io/Vitis_Libraries/graph/2021.1/index.html).

## License

Licensed using the [Apache 2.0 license](https://www.apache.org/licenses/LICENSE-2.0).

    Copyright 2019 Xilinx, Inc.
    
    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at
    
        http://www.apache.org/licenses/LICENSE-2.0
    
    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.
    Copyright 2019 Xilinx, Inc.



