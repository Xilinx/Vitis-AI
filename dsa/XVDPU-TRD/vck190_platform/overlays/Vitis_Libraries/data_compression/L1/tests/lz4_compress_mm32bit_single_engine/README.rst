Xilinx LZ4 32 Bit Memory Mapped Single Engine Compress HLS Test
===============================================================

**Description:** This is a L1 test design to validate LZ4 compression module. It processes the data to and from the DDR into multiple parallel streams that helps in processing 8x data and achieve higher performance.

**Top Function:** hls_lz4CompressMM32bitSingleEngine

Results
-------

==================== ===== ===== ==== ==== 
Module               LUT   FF    BRAM URAM 
lz4_compress_test    6110  8247  13   6 
==================== ===== ===== ==== ==== 