// RUN: tf-mlir-translate -mlir-to-graphdef %s -o - | FileCheck %s

func @main() {
// CHECK:      node {
// CHECK-NEXT:   name: "predicate"
// CHECK-NEXT:   op: "Const"
// CHECK-NEXT:   attr {
// CHECK-NEXT:     key: "dtype"
// CHECK-NEXT:     value {
// CHECK-NEXT:       type: DT_INT32
// CHECK-NEXT:     }
// CHECK-NEXT:   }
// CHECK-NEXT:   attr {
// CHECK-NEXT:     key: "value"
// CHECK-NEXT:     value {
// CHECK-NEXT:       tensor {
// CHECK-NEXT:         dtype: DT_INT32
// CHECK-NEXT:         tensor_shape {
// CHECK-NEXT:         }
// CHECK-NEXT:         int_val: 0
// CHECK-NEXT:       }
// CHECK-NEXT:     }
// CHECK-NEXT:   }
// CHECK:      }
  %0:2 = "_tf.Const"() {device = "", dtype = "tfdtype$DT_INT32", value = dense<0> : tensor<i32>} : () -> (tensor<i32>, !_tf.control) loc("predicate")

// CHECK:      node {
// CHECK-NEXT:   name: "Case"
// CHECK-NEXT:   op: "Case"
// CHECK-NEXT:   input: "predicate"
// CHECK:        attr {
// CHECK:          key: "branches"
// CHECK-NEXT:     value {
// CHECK-NEXT:       list {
// CHECK-NEXT:         func {
// CHECK-NEXT:           name: "foo"
// CHECK-NEXT:         }
// CHECK-NEXT:         func {
// CHECK-NEXT:           name: "bar"
// CHECK-NEXT:         }
// CHECK-NEXT:       }
// CHECK-NEXT:     }
// CHECK-NEXT:   }
// CHECK:      }
  %1:2 = "_tf.Case"(%0#0) {Tin = [], Tout = ["tfdtype$DT_FLOAT"], branches = [@foo, @bar], device = "", output_shapes = []} : (tensor<i32>) -> (tensor<*xf32>, !_tf.control) loc("Case")
  return
}

// CHECK-DAG: name: "foo"
func @foo() -> tensor<10xf32> {
  %0:2 = "_tf.Const"() {device = "", dtype = "tfdtype$DT_FLOAT", value = dense<1.000000e+00> : tensor<10xf32>} : () -> (tensor<10xf32>, !_tf.control) loc("const_1")
  return %0#0 : tensor<10xf32>
}

// CHECK-DAG: name: "bar"
func @bar() -> tensor<10xf32> {
  %0:2 = "_tf.Const"() {device = "", dtype = "tfdtype$DT_FLOAT", value = dense<2.000000e+00> : tensor<10xf32>} : () -> (tensor<10xf32>, !_tf.control) loc("const_2")
  return %0#0 : tensor<10xf32>
}
