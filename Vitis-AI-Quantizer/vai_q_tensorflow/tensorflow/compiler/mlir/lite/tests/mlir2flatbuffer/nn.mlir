// RUN: flatbuffer_translate -mlir-to-tflite-flatbuffer %s -o - | flatbuffer_to_string - | FileCheck %s

func @main(tensor<1x6x6x16xf32>) -> tensor<1x1x1x16xf32> {
^bb0(%arg0: tensor<1x6x6x16xf32>):
  // CHECK:      {
  // CHECK-NEXT:   version: 3,
  // CHECK-NEXT:   operator_codes: [ {
  // CHECK-NEXT:     builtin_code: AVERAGE_POOL_2D
  // CHECK-NEXT:   } ],
  // CHECK-NEXT:   subgraphs: [ {
  // CHECK-NEXT:     tensors: [ {
  // CHECK-NEXT:       shape: [ 1, 6, 6, 16 ],
  // CHECK-NEXT:       buffer: 1,
  // CHECK-NEXT:       name: "Input",
  // CHECK-NEXT:       quantization: {
  // CHECK-EMPTY:
  // CHECK-NEXT:       }
  // CHECK-NEXT:     }, {
  // CHECK-NEXT:       shape: [ 1, 1, 1, 16 ],
  // CHECK-NEXT:       buffer: 2,
  // CHECK-NEXT:       name: "avgpool",
  // CHECK-NEXT:       quantization: {
  // CHECK-EMPTY:
  // CHECK-NEXT:       }
  // CHECK-NEXT:     } ],
  // CHECK-NEXT:     inputs: [ 0 ],
  // CHECK-NEXT:     outputs: [ 1 ],
  // CHECK-NEXT:     operators: [ {
  // CHECK-NEXT:       inputs: [ 0 ],
  // CHECK-NEXT:       outputs: [ 1 ],
  // CHECK-NEXT:       builtin_options_type: Pool2DOptions,
  // CHECK-NEXT:       builtin_options: {
  // CHECK-NEXT:         padding: VALID,
  // CHECK-NEXT:         stride_w: 1,
  // CHECK-NEXT:         stride_h: 3,
  // CHECK-NEXT:         filter_width: 6,
  // CHECK-NEXT:         filter_height: 3
  // CHECK-NEXT:       }
  // CHECK-NEXT:     } ]
  // CHECK-NEXT:    name: "main"
  // CHECK-NEXT:   } ],
  // CHECK-NEXT:   description: "MLIR Converted.",
  // CHECK-NEXT:   buffers: [ {
  // CHECK-EMPTY:
  // CHECK-NEXT:   }, {
  // CHECK-EMPTY:
  // CHECK-NEXT:   }, {
  // CHECK-EMPTY:
  // CHECK-NEXT:   } ]
  // CHECK-NEXT: }

  %0 = "tfl.pseudo_input"(%arg0) : (tensor<1x6x6x16xf32>) -> tensor<1x6x6x16xf32> loc("Input")
  %1 = "tfl.average_pool_2d"(%0) {filter_height = 3 : i32, filter_width = 6 : i32, fused_activation_function = "NONE", padding = "VALID", stride_h = 3 : i32, stride_w = 1 : i32} : (tensor<1x6x6x16xf32>) -> tensor<1x1x1x16xf32> loc("avgpool")
  return %1 : tensor<1x1x1x16xf32>
}
