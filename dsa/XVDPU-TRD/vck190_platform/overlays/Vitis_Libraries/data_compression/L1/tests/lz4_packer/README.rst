Xilinx LZ4 Packer HLS Test
==========================

**Description:** This is a L1 test design to validate LZ4 compression and packer module. It processes the data to and from the DDR into multiple parallel streams that helps in processing 8x data and achieve higher performance and performs header processing to give an output file.

**Top Function:** hls_lz4CompressPacker

Results
-------

==================== ===== ===== ==== ==== 
Module               LUT   FF    BRAM URAM 
lz4_packer_test      23449 25948 43   6 
==================== ===== ===== ==== ==== 