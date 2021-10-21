# TigerGraph Integration

TigerGraph is a complete, distributed, parallel graph computing platform supporting web-scale data analytics in real-time. GSQL is a language designed for TigerGraph Inc.'s property graph database. Users can define their own expression functions in C++ and add them into GSQL. Graph L3 APIs can be easily integrated with TigerGraph.

## Requirements
### Software Requirements
- Ubuntu 16.04 LTS
- [Xilinx RunTime (XRT) 2020.1](https://github.com/Xilinx/XRT)
- [Xilinx FPGA Resource Manager (XRM) 2020.2](https://github.com/Xilinx/XRM)

The integration needs static boost version XRT. Please follow the build steps in [Vitis Graph Library Documentation](https://xilinx.github.io/Vitis_Libraries/graph/2020.1/plugin/tigergraph_integration.html).

### Hardware Requirements
- [Alveo U50](https://www.xilinx.com/products/boards-and-kits/alveo/u50.html)

## Integration Flow
In order to simplify the integration of graph L3 and TigerGraph, a shell script is written. Please follow the steps in [Vitis Graph Library Documentation](https://xilinx.github.io/Vitis_Libraries/graph/2020.1/plugin/tigergraph_integration.html).

