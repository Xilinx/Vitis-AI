// RUN: flatbuffer_translate -mlir-to-tflite-flatbuffer %s -o - | flatbuffer_to_string - | FileCheck --dump-input-on-failure %s
// RUN: flatbuffer_translate -mlir-to-tflite-flatbuffer %s -o - -strip-debug-info | flatbuffer_to_string - | FileCheck --dump-input-on-failure %s --check-prefix=STRIP

func @main(tensor<3x2xi32>) -> tensor<3x2xi32>
  attributes {tf.entry_function = {inputs = "input", outputs = "SameNameAsOutput"}} {
^bb0(%arg0: tensor<3x2xi32>):
// CHECK: {
// CHECK-NEXT:   version: 3,
// CHECK-NEXT:   operator_codes: [ {
// CHECK-NEXT:     builtin_code: SUB
// CHECK-NEXT:   }, {
// CHECK-EMPTY:
// CHECK-NEXT:   } ],
// CHECK-NEXT:   subgraphs: [ {
// CHECK-NEXT:     tensors: [ {
// CHECK-NEXT:       shape: [ 3, 2 ],
// CHECK-NEXT:       type: INT32,
// CHECK-NEXT:       buffer: 1,
// CHECK-NEXT:       name: "input",
// STRIP:            buffer: 1,
// STRIP-NEXT:       name: "input",
// CHECK-NEXT:       quantization: {
// CHECK-EMPTY:
// CHECK-NEXT:       }
// CHECK-NEXT:     }, {
// CHECK-NEXT:       shape: [ 3, 2 ],
// CHECK-NEXT:       type: INT32,
// CHECK-NEXT:       buffer: 2,
// CHECK-NEXT:       name: "Const",
// STRIP:            buffer: 2,
// STRIP-NEXT:       name: "0",
// CHECK-NEXT:       quantization: {
// CHECK-EMPTY:
// CHECK-NEXT:       }
// CHECK-NEXT:     }, {
// CHECK-NEXT:       shape: [ 3, 2 ],
// CHECK-NEXT:       type: INT32,
// CHECK-NEXT:       buffer: 3,
// CHECK-NEXT:       name: "sub",
// STRIP:            buffer: 3,
// STRIP-NEXT:       name: "1",
// CHECK-NEXT:       quantization: {
// CHECK-EMPTY:
// CHECK-NEXT:       }
// CHECK-NEXT:     }, {
// CHECK-NEXT:       shape: [ ],
// CHECK-NEXT:       type: INT32,
// CHECK-NEXT:       buffer: 4,
// CHECK-NEXT:       name: "SameNameAsOutput1",
// STRIP:            buffer: 4,
// STRIP-NEXT:       name: "2",
// CHECK-NEXT:       quantization: {
// CHECK-EMPTY:
// CHECK-NEXT:       }
// CHECK-NEXT:     }, {
// CHECK-NEXT:       shape: [ 3, 2 ],
// CHECK-NEXT:       type: INT32,
// CHECK-NEXT:       buffer: 5,
// CHECK-NEXT:       name: "SameNameAsOutput",
// STRIP:            buffer: 5,
// STRIP-NEXT:       name: "SameNameAsOutput",
// CHECK-NEXT:       quantization: {
// CHECK-EMPTY:
// CHECK-NEXT:       }
// CHECK-NEXT:     } ],
// CHECK-NEXT:     inputs: [ 0 ],
// CHECK-NEXT:     outputs: [ 4 ],
// CHECK-NEXT:     operators: [ {
// CHECK-NEXT:       inputs: [ 0, 1 ],
// CHECK-NEXT:       outputs: [ 2 ],
// CHECK-NEXT:       builtin_options_type: SubOptions,
// CHECK-NEXT:       builtin_options: {
// CHECK-NEXT:         fused_activation_function: RELU6
// CHECK-NEXT:       }
// CHECK-NEXT:     }, {
// CHECK-NEXT:       opcode_index: 1,
// CHECK-NEXT:       inputs: [ 3, 2 ],
// CHECK-NEXT:       outputs: [ 4 ],
// CHECK-NEXT:       builtin_options_type: AddOptions,
// CHECK-NEXT:       builtin_options: {
// CHECK-EMPTY:
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
// CHECK-NEXT:     data: [ 1, 0, 0, 0, 2, 0, 0, 0, 3, 0, 0, 0, 4, 0, 0, 0, 5, 0, 0, 0, 6, 0, 0, 0 ]
// CHECK-NEXT:   }, {
// CHECK-EMPTY:
// CHECK-NEXT:   }, {
// CHECK-NEXT:     data: [ 10, 0, 0, 0 ]
// CHECK-NEXT:   }, {
// CHECK-EMPTY:
// CHECK-NEXT:   } ]
// CHECK-NEXT: }

  %0 = "tfl.pseudo_input" (%arg0) : (tensor<3x2xi32>) -> tensor<3x2xi32> loc("Input")
  %1 = "tfl.pseudo_const" () {value = dense<[[1, 2], [3, 4], [5, 6]]> : tensor<3x2xi32>} : () -> tensor<3x2xi32> loc("Const")
  %2 = "tfl.sub" (%0, %1) {fused_activation_function = "RELU6"} : (tensor<3x2xi32>, tensor<3x2xi32>) -> tensor<3x2xi32> loc("sub")
  %3 = "std.constant" () {value = dense<10> : tensor<i32>} : () -> tensor<i32> loc("SameNameAsOutput")
  %4 = "tfl.add" (%3, %2) {fused_activation_function = "NONE"} : (tensor<i32>, tensor<3x2xi32>) -> tensor<3x2xi32> loc("add")
  return %4 : tensor<3x2xi32>
}
