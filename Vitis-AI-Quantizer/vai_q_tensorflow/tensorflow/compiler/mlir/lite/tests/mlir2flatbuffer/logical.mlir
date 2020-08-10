// RUN: flatbuffer_translate -mlir-to-tflite-flatbuffer %s -o - | flatbuffer_to_string - | FileCheck %s

func @main(tensor<4xi1>) -> tensor<4xi1> {
^bb0(%arg0: tensor<4xi1>):
  // CHECK:      {
  // CHECK-NEXT:   version: 3,
  // CHECK-NEXT:   operator_codes: [ {
  // CHECK-NEXT:     builtin_code: LOGICAL_OR
  // CHECK-NEXT:   }, {
  // CHECK-NEXT:     builtin_code: LOGICAL_AND
  // CHECK-NEXT:   } ],
  // CHECK-NEXT:   subgraphs: [ {
  // CHECK-NEXT:     tensors: [ {
  // CHECK-NEXT:       shape: [ 4 ],
  // CHECK-NEXT:       type: BOOL,
  // CHECK-NEXT:       buffer: 1,
  // CHECK-NEXT:       name: "Input",
  // CHECK-NEXT:       quantization: {
  // CHECK-EMPTY:
  // CHECK-NEXT:       }
  // CHECK-NEXT:     }, {
  // CHECK-NEXT:       shape: [ 4 ],
  // CHECK-NEXT:       type: BOOL,
  // CHECK-NEXT:       buffer: 2,
  // CHECK-NEXT:       name: "Const1",
  // CHECK-NEXT:       quantization: {
  // CHECK-EMPTY:
  // CHECK-NEXT:       }
  // CHECK-NEXT:     }, {
  // CHECK-NEXT:       shape: [ 4 ],
  // CHECK-NEXT:       type: BOOL,
  // CHECK-NEXT:       buffer: 3,
  // CHECK-NEXT:       name: "Const2",
  // CHECK-NEXT:       quantization: {
  // CHECK-EMPTY:
  // CHECK-NEXT:       }
  // CHECK-NEXT:     }, {
  // CHECK-NEXT:       shape: [ 4 ],
  // CHECK-NEXT:       type: BOOL,
  // CHECK-NEXT:       buffer: 4,
  // CHECK-NEXT:       name: "logical_or",
  // CHECK-NEXT:       quantization: {
  // CHECK-EMPTY:
  // CHECK-NEXT:       }
  // CHECK-NEXT:     }, {
  // CHECK-NEXT:       shape: [ 4 ],
  // CHECK-NEXT:       type: BOOL,
  // CHECK-NEXT:       buffer: 5,
  // CHECK-NEXT:       name: "logical_and",
  // CHECK-NEXT:       quantization: {
  // CHECK-EMPTY:
  // CHECK-NEXT:       }
  // CHECK-NEXT:     } ],
  // CHECK-NEXT:     inputs: [ 0 ],
  // CHECK-NEXT:     outputs: [ 4 ],
  // CHECK-NEXT:     operators: [ {
  // CHECK-NEXT:       inputs: [ 0, 2 ],
  // CHECK-NEXT:       outputs: [ 3 ]
  // CHECK-NEXT:     }, {
  // CHECK-NEXT:       opcode_index: 1,
  // CHECK-NEXT:       inputs: [ 3, 1 ],
  // CHECK-NEXT:       outputs: [ 4 ]
  // CHECK-NEXT:     } ]
  // CHECK-NEXT:    name: "main"
  // CHECK-NEXT:   } ],
  // CHECK-NEXT:   description: "MLIR Converted.",
  // CHECK-NEXT:   buffers: [ {
  // CHECK-EMPTY:
  // CHECK-NEXT:   }, {
  // CHECK-EMPTY:
  // CHECK-NEXT:   }, {
  // CHECK-NEXT:     data: [ 1, 1, 1, 1 ]
  // CHECK-NEXT:   }, {
  // CHECK-NEXT:     data: [ 0, 0, 0, 0 ]
  // CHECK-NEXT:   }, {
  // CHECK-EMPTY:
  // CHECK-NEXT:   }, {
  // CHECK-EMPTY:
  // CHECK-NEXT:   } ]
  // CHECK-NEXT: }
  // CHECK-EMPTY:

  %0 = "tfl.pseudo_input" (%arg0) : (tensor<4xi1>) -> tensor<4xi1> loc("Input")
  %1 = "tfl.pseudo_const" () {value = dense<true> : tensor<4xi1>} : () -> tensor<4xi1> loc("Const1")
  %2 = "tfl.pseudo_const" () {value = dense<false> : tensor<4xi1>} : () -> tensor<4xi1> loc("Const2")
  %3 = "tfl.logical_or"(%0, %2) : (tensor<4xi1>, tensor<4xi1>) -> tensor<4xi1> loc("logical_or")
  %4 = "tfl.logical_and"(%3, %1) : (tensor<4xi1>, tensor<4xi1>) -> tensor<4xi1> loc("logical_and")
  return %4 : tensor<4xi1>
}
