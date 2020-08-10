
from .option_def import Option

####################################
# Add Option list here
####################################


class NndctOption(object):
  nndct_help = Option(name="help", dtype=bool, default=False, action="store_true", 
                      help="list all api usage description")

  nndct_quant_off = Option(name="quant_off", dtype=bool, default=False, action="store_true", 
                           help="disable quantization flow")

  nndct_option_list = Option(name="option_list", dtype=bool, default=False, action="store_true", 
                             help="list all the options in nndct")

  nndct_parse_debug = Option(name="parse_debug", dtype=int, default=0, 
                             help="logging graph, 1: torch raw graph, 2: nndct graph")

  nndct_logging_level = Option(name="logging_level", dtype=int, default=0, help="logging level")
  
  nndct_quant_mode = Option(name="quant_mode", dtype=int, default=0, 
                            help="quant mode, 1:calibration, 2:quantization")
  
  nndct_quant_mode = Option(name="dump_float_format", dtype=int, default=0, 
                            help="deploy check data format, 0: bin, 1: txt")