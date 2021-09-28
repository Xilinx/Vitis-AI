# Key L1 Module Benchmark Test

Many database key operations are heavy on memory access, and as a result most key L1 modules
in this library has AXI-master port to store intermediate data structure or partial result in DRAM.
While latency in DRAM dominate these primitives, it cannot be accurately modeled in HLS simulation.

Therefore in this benchmark folder, key L1 modules are combined with fast data source and sink into Vitis kernels, 
as these kernels can enable more-accurate Vitis hardware emulation or direct run on Alveo card for performance evaluation.
