Xilinx Lz4 Multistream Compress HLS Test
========================================

**Description:** This is a L1 test design to validate LZ4 compression module. It processes the data to and from the DDR into multiple parallel streams that helps in processing 8x data and achieve higher performance.

**Top Function:** hls_lz4CompressMutipleStreams

Results
-------

==================== ===== ===== ==== ==== 
Module               LUT   FF    BRAM URAM 
lz4_compress_test    8933  14391 42   6 
==================== ===== ===== ==== ==== 