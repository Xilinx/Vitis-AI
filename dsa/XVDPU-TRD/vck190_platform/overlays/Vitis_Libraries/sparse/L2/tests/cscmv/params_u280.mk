#+-------------------------------------------------------------------------------
# The following parameters are assigned with default values. These parameters can
# be overridden through the make command line
#+-------------------------------------------------------------------------------
SPARSE_maxParamDdrBlocks=1024
SPARSE_maxParamHbmBlocks=512
SPARSE_paramOffset=1024
SPARSE_maxColMemBlocks = 128 
SPARSE_maxColParBlocks = 512 
SPARSE_maxRowBlocks = 512 
SPARSE_dataType = float 
SPARSE_indexType = uint32_t
SPARSE_logParEntries = 2
SPARSE_parEntries = 4
SPARSE_logParGroups = 0 
SPARSE_parGroups = 1
SPARSE_dataBits = 32
SPARSE_indexBits = 32
SPARSE_hbmMemBits = 256
SPARSE_ddrMemBits = 512
SPARSE_hbmChannels = 16 
SPARSE_hbmChannelMegaBytes = 256
SPARSE_printWidth = 6
SPARSE_pageSize = 4096
DEBUG_dumpData=0
SEQ_KERNEL=0

COMMON_DEFS = -D SPARSE_maxParamDdrBlocks=$(SPARSE_maxParamDdrBlocks) \
				-D SPARSE_maxParamHbmBlocks=$(SPARSE_maxParamHbmBlocks) \
				-D SPARSE_paramOffset=$(SPARSE_paramOffset) \
				-D SPARSE_maxColMemBlocks=$(SPARSE_maxColMemBlocks) \
				-D SPARSE_maxColParBlocks=$(SPARSE_maxColParBlocks) \
				-D SPARSE_maxRowBlocks=$(SPARSE_maxRowBlocks) \
				-D SPARSE_dataType=$(SPARSE_dataType) \
				-D SPARSE_indexType=$(SPARSE_indexType) \
				-D SPARSE_logParEntries=$(SPARSE_logParEntries) \
				-D SPARSE_parEntries=$(SPARSE_parEntries) \
				-D SPARSE_logParGroups=$(SPARSE_logParGroups) \
				-D SPARSE_parGroups=$(SPARSE_parGroups) \
				-D SPARSE_dataBits=$(SPARSE_dataBits) \
				-D SPARSE_indexBits=$(SPARSE_indexBits) \
				-D SPARSE_hbmMemBits=$(SPARSE_hbmMemBits) \
				-D SPARSE_ddrMemBits=$(SPARSE_ddrMemBits) \
				-D SPARSE_printWidth=$(SPARSE_printWidth) \
				-D SPARSE_pageSize=$(SPARSE_pageSize) \
				-D SPARSE_hbmChannels=$(SPARSE_hbmChannels) \
				-D SPARSE_hbmChannelMegaBytes=$(SPARSE_hbmChannelMegaBytes) \
				-D DEBUG_dumpData=$(DEBUG_dumpData) \
				-D SEQ_KERNEL=$(SEQ_KERNEL)

GEN_PARTITION_DEFS = $(COMMON_DEFS) 
