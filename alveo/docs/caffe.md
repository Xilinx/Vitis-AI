user@localhost:/workspace$ conda activate vitis-ai-caffe

(vitis-ai-caffe) user@localhost:/workspace$ vai_q_caffe --help

vai_q_caffe: command line brew

usage: vai_q_caffe <command> <args>

commands:

     quantize        quantize a float model (need calibration with dataset) and deploy to DPU
  
     deploy          deploy quantized model to DPU
  
     finetune        train or finetune a model
  
     test            score a model

example:
  1. quantize:                           vai_q_caffe quantize -model float.prototxt -weights float.caffemodel -gpu 0
  2. quantize with auto test:            vai_q_caffe quantize -model float.prototxt -weights float.caffemodel -gpu 0 -auto_test -test_iter 50
  3. quantize with Non-Overflow method:  vai_q_caffe quantize -model float.prototxt -weights float.caffemodel -gpu 0 -method 0
  4. finetune quantized model:           vai_q_caffe finetune -solver solver.prototxt -weights quantize_results/float_train_test.caffemodel -gpu 0
  5. deploy quantized model:             vai_q_caffe depoly -model quantize_results/quantize_train_test.prototxt -weights quantize_results/float_train_test.caffemodel -gpu 0


  Available options:
  
    -ap_style (Optional: AP computing styles for detection net, including:
      11point, MaxIntegral, Integral) type: string default: "11point"
      
    -auto_test (Optional: Whether to test after calibration, need test dataset)
      type: bool default: false
      
    -calib_iter (Optional: Total iterations for quantize calibration)
      type: int32 default: 100
      
    -data_bit (Optional: Bit width for activation data during quantization)
      type: int32 default: 8
      
    -gpu (Optional: The GPU device id for calibration and test) type: string
      default: ""
      
    -ignore_layers (Optinal: List of layers to be ignore during quantization,
      comma-delimited) type: string default: ""
      
    -ignore_layers_file (Optional: Prototxt file which defines the layers to be
      ignore during quantization, each line formated as 'ignore_layer:
      "layer_name"') type: string default: ""
      
    -input_blob (Optional: Name of input data blob) type: string
      default: "data"
      
    -keep_fixed_neuron (Remain FixedNeuron layers in the output float
      caffemodel ) type: bool default: false
      
    -method (Optional: Method for quantization, 0: Non-Overflow, 1: Min-Diff)
      type: int32 default: 1
      
    -model (The model definition protocol buffer text file) type: string
      default: ""
      
    -net_type (Optional: Net types for net testing, including: classification,
      detection, segmentation, other) type: string default: ""
      
    -output_dir (Optional: Output directory for the quantization results)
      type: string default: "./quantize_results"
      
    -seg_class_num (Optional; The number of segmentation classes) type: int32
      default: 19
      
    -seg_metric (Optional; Test metric for segmentation net, now only classIOU
      is supported) type: string default: "classIOU"
      
    -sigmoided_layers (Optinal: List of layers before sigmoid operation, to be
      quantized with optimization for sigmoid accuracy) type: string
      default: ""
      
    -solver (The solver for finetuning or training) type: string default: ""
    
    -special_layers (Optinal: List of layers which has special data_bit or
      weight_bit for quantization, comma-delimited, e.g.
      'data_in,16,conv2,8,16', note:(1) the first number after layer_name
      represent data_bit, and the second number represent weight_bit(optional,
      can be omitted). (2) We have take some necessary measures to check the
      input parameters) type: string default: ""
      
    -test_iter (Optional: Total iterations for test) type: int32 default: 50
    
    -weights (The pretrained float weights for quantization) type: string
      default: ""
      
    -weights_bit (Optional: Bit width for weights and bias during quantization)
      type: int32 default: 8
