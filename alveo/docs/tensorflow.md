user@localhost:/workspace$ conda activate vitis-ai-tensorflow

(vitis-ai-tensorflow) user@localhost:/workspace$ vai_q_tensorflow --help

usage: 

    usage: vai_q_tensorflow command [Options]
    
    examples:
      show help       : vai_q_tensorflow --help
      quantize a model: vai_q_tensorflow quantize --input_frozen_graph frozen_graph.pb --input_nodes xxx --output_nodes yyy --input_shapes zzz --input_fn module.calib_input
      inspect a model : vai_q_tensorflow inspect --input_frozen_graph frozen_graph.pb
      dump quantized model : vai_q_tensorflow dump --input_frozen_graph quantize_results/quantize_eval_model.pb --input_fn module.dump_input
  

Xilinx's Quantization Tools vai_q_tensorflow v0.3.1 build for tensorflow 1.12.0

positional arguments:
  {quantize,deploy,inspect,dump}
                        Specify a command for decent_q

optional arguments:

     -h, --help            show this help message and exit
  
     --version             show program's version number and exit
  
     --mode {frozen,train,eval}
                        Mode for quantization. (default: frozen)
                        
     --input_frozen_graph INPUT_FROZEN_GRAPH
                        Path to input frozen graph(.pb) (default: )
                        
     --input_meta_graph INPUT_META_GRAPH
                        Path to input meta graph(.meta) (default: )
                        
     --input_checkpoint INPUT_CHECKPOINT
                        Path to checkpoint files (default: )
                        
     --output_dir OUTPUT_DIR
                        The directory to save the quantization results
                        (default: ./quantize_results)
                        
     --weight_bit WEIGHT_BIT
                        The bit width for weights/biases (default: 8)
                        
     --activation_bit ACTIVATION_BIT
                        The bit width for activation (default: 8)
                        
     --method {0,1}        The method for quantization, options are: 1) 0: non-
                        overflow method, make sure no values are saturated
                        during quantization, may get worse results incase of
                        outliers. 2) 1: min-diffs method, allow saturation for
                        quantization to get lower quantization difference,
                        higher endurance to outliers. Usually end with
                        narrower ranges than non-overflow method. (default: 1)
                        
     --calib_iter CALIB_ITER
                        The iterations of calibration, total number of images
                        for calibration = calib_iter * batch_size (default:
                        100)
                        
     --input_nodes INPUT_NODES
                        The name list of input nodes of the subgraph to be
                        quantized, comma separated. Used together with
                        output_nodes. If not set, the whole graph between
                        input_nodes and output nodes will be quantized. When
                        generating the model for deploy, only the subgraph
                        between input_nodes and output_nodes will be included.
                        This parameter is often used when only want to
                        quantize part of the graph, such as the main body of
                        the model without preprocessing and postprocessing.
                        (default: )
                        
     --input_shapes INPUT_SHAPES
                        The shape list of input_nodes, If not set, the 'shape'
                        from the NodeDef will be used. The shape must be a
                        4-dimension shape for each node,, comma separated,
                        e.g. 1,224,224,3; Unknown size for batchsize is
                        supported, e.g. ?,224,224,3; In case of multiple
                        input_nodes, please assign shape list of each node,
                        separated by `:`. e.g. ?,224,224,3:?,300,300,1
                        (default: )
                        
     --output_nodes OUTPUT_NODES
                        The name list of output nodes of the subgraph to be
                        quantized, comma separated. Used together with
                        input_nodes. If not set, the whole graph between
                        input_nodes and output nodes will be quantized. When
                        generating the model for deploy, only the subgraph
                        between input_nodes and output_nodes will be included.
                        This parameter is often used when only want to
                        quantize part of the graph, such as the main body of
                        the model without preprocessing and postprocessing.
                        (default: )
                        
     --ignore_nodes IGNORE_NODES
                        The name list of nodes to be ignored during
                        quantization, comma separated. ignored nodes will be
                        left unquantized during quantization. (default: )
                        
     --skip_check {0,1}    If set 1, the check for float model will be skipped,
                        useful when only part of the input model is quantized.
                        (default: 0)
                        
     --align_concat {0,1,2}
                        The strategy for alignment of the input quantize
                        position for concat nodes. Set 0 to align all concat
                        nodes, 1 to align the output concat nodes, 2 to
                        disable alignment (default: 0)
                        
     --simulate_dpu {0,1}  Set to 1 to enable simulation of DPU. The behavior of
                        DPU for some operations are different from tensorflow.
                        For example, the dividing in LeakyRelu and AvgPooling
                        are replaced by bit-shifting, so there maybe slight
                        difference between DPU outputs and CPU/GPU outputs.
                        Decent_q will simulate the behavior for these
                        operations if this flag is set to 1 (default: 1)
                        
     --input_fn INPUT_FN   The function that provides input data for graph, the
                        format function format is `model_name.input_fn_name`,
                        e.g. 'my_input_fn.input_fn'. The input_fn should take
                        a `int` object as input indicating the calibration
                        step, and should return a dict`(input_node_name,
                        numpy.Array)` object for each call, which will be fed
                        into the model's placeholder operations. (default: )
                        
     --max_dump_batches MAX_DUMP_BATCHES
                        The maximum batches to be dumped (default: 1)
                        
     --dump_float {0,1}    If set 1, the float weights and activation tensors
                        will also be dumped. (default: 0)
                        
     --gpu GPU             The gpu id used for quantization, comma separated.
                        (default: 0)
                        
     --gpu_memory_fraction GPU_MEMORY_FRACTION
                        The gpu memory fraction used for quantization, between
                        0-1. (default: 0.5)
