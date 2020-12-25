// RUN: flatbuffer_translate -mlir-to-tflite-flatbuffer %s -emit-select-tf-ops -o - | flatbuffer_to_string - | FileCheck %s

func @main(tensor<4xf32>) -> tensor<4xf32> {
^bb0(%arg0: tensor<4xf32>):
// CHECK:  {
// CHECK-NEXT:  version: 3,
// CHECK-NEXT:  operator_codes: [ {
// CHECK-NEXT:    builtin_code: MUL
// CHECK-NEXT:  }, {
// CHECK-NEXT:    builtin_code: CUSTOM,
// CHECK-NEXT:    custom_code: "FlexDiv"
// CHECK-NEXT:  }, {
// CHECK-NEXT:    builtin_code: EXP
// CHECK-NEXT:  } ],
// CHECK-NEXT:  subgraphs: [ {
// CHECK-NEXT:    tensors: [ {
// CHECK-NEXT:      shape: [ 4 ],
// CHECK-NEXT:      buffer: 1,
// CHECK-NEXT:      name: "Input",
// CHECK-NEXT:      quantization: {
// CHECK-EMPTY:
// CHECK-NEXT:      }
// CHECK-NEXT:    }, {
// CHECK-NEXT:      shape: [ 4 ],
// CHECK-NEXT:      buffer: 2,
// CHECK-NEXT:      name: "Const",
// CHECK-NEXT:      quantization: {
// CHECK-EMPTY:
// CHECK-NEXT:      }
// CHECK-NEXT:    }, {
// CHECK-NEXT:      shape: [ 4 ],
// CHECK-NEXT:      buffer: 3,
// CHECK-NEXT:      name: "mul",
// CHECK-NEXT:      quantization: {
// CHECK-EMPTY:
// CHECK-NEXT:      }
// CHECK-NEXT:    }, {
// CHECK-NEXT:      shape: [ 4 ],
// CHECK-NEXT:      buffer: 4,
// CHECK-NEXT:      name: "div",
// CHECK-NEXT:      quantization: {
// CHECK-EMPTY:
// CHECK-NEXT:      }
// CHECK-NEXT:    }, {
// CHECK-NEXT:      shape: [ 4 ],
// CHECK-NEXT:      buffer: 5,
// CHECK-NEXT:      name: "exp",
// CHECK-NEXT:      quantization: {
// CHECK-EMPTY:
// CHECK-NEXT:      }
// CHECK-NEXT:    } ],
// CHECK-NEXT:    inputs: [ 0 ],
// CHECK-NEXT:    outputs: [ 4 ],
// CHECK-NEXT:    operators: [ {
// CHECK-NEXT:      inputs: [ 0, 1 ],
// CHECK-NEXT:      outputs: [ 2 ],
// CHECK-NEXT:      builtin_options_type: MulOptions,
// CHECK-NEXT:      builtin_options: {
// CHECK-EMPTY:
// CHECK-NEXT:      }
// CHECK-NEXT:    }, {
// CHECK-NEXT:      opcode_index: 1,
// CHECK-NEXT:      inputs: [ 2, 1 ],
// CHECK-NEXT:      outputs: [ 3 ],
// CHECK-NEXT:      custom_options: [ 3, 68, 105, 118, 0, 20, 18, 3, 68, 105, 118, 26, 0, 26, 0, 42, 7, 10, 1, 84, 18, 2, 48, 1, 50, 0, 0, 2, 27, 23, 20, 20, 4, 40, 1 ]
// CHECK-NEXT:    }, {
// CHECK-NEXT:      opcode_index: 2,
// CHECK-NEXT:      inputs: [ 3 ],
// CHECK-NEXT:      outputs: [ 4 ],
// CHECK-NEXT:      builtin_options_type: ExpOptions,
// CHECK-NEXT:      builtin_options: {
// CHECK-EMPTY:
// CHECK-NEXT:      }
// CHECK-NEXT:    } ]
// CHECK-NEXT:    name: "main"
// CHECK-NEXT:  } ],
// CHECK-NEXT:  description: "MLIR Converted.",
// CHECK-NEXT:  buffers: [ {
// CHECK-EMPTY:
// CHECK-NEXT:  }, {
// CHECK-EMPTY:
// CHECK-NEXT:  }, {
// CHECK-NEXT:    data: [ 0, 0, 128, 63, 0, 0, 128, 63, 0, 0, 128, 63, 0, 0, 128, 63 ]
// CHECK-NEXT:  }, {
// CHECK-EMPTY:
// CHECK-NEXT:  }, {
// CHECK-EMPTY:
// CHECK-NEXT:  }, {
// CHECK-EMPTY:
// CHECK-NEXT:  } ]
// CHECK-NEXT:}

  %0 = "tfl.pseudo_input" (%arg0)  : (tensor<4xf32>) -> tensor<4xf32> loc("Input")
  %1 = "tfl.pseudo_const" () {value = dense<1.0> : tensor<4xf32>} : () -> tensor<4xf32> loc("Const")
  %2 = "tfl.mul"(%0, %1) {fused_activation_function = "NONE"} : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32> loc("mul")
  // tf.div is the result of conversion to a Flex TF op
  %3 = "tf.Div"(%2, %1)  : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32> loc("div")
  %4 = "tfl.exp"(%3)  : (tensor<4xf32>) -> tensor<4xf32> loc("exp")
  return %4 : tensor<4xf32>
}
