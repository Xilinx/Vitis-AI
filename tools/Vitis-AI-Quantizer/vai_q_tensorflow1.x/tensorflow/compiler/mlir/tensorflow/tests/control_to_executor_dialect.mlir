// RUN: tf-opt -tf-control-to-executor-conversion %s  | FileCheck %s --dump-input=fail


// CHECK-LABEL: func @islands_with_control
// CHECK-SAME: (%[[ARG0:[a-z0-9]*]]: tensor<*xf32>)
func @islands_with_control(tensor<*xf32>) -> tensor<*xf32> {
^bb0(%0: tensor<*xf32>):
  %1:2 = "_tf.Identity"(%0) : (tensor<*xf32>) -> (tensor<*xf32>, !_tf.control)
  %2 = "_tf.Add"(%0, %0, %1#1) : (tensor<*xf32>, tensor<*xf32>, !_tf.control) -> tensor<*xf32>
  return %2 : tensor<*xf32>
}

// CHECK-NEXT: %[[GRAPH:[0-9]*]] = tf_executor.graph {
// CHECK-NEXT:   %[[IDENTITY:[0-9]*]]:2 = tf_executor.island {
// CHECK-NEXT:     %{{[0-9]*}} = "tf.Identity"(%[[ARG0]]) : (tensor<*xf32>) -> tensor<*xf32>
// CHECK-NEXT:     tf_executor.yield %{{[0-9]*}} : tensor<*xf32>
// CHECK-NEXT:   }
// CHECK-NEXT:   %[[ADD:[0-9]*]]:2 = tf_executor.island(%[[IDENTITY]]#1) {
// CHECK-NEXT:     %{{[0-9]*}} = "tf.Add"(%[[ARG0]], %[[ARG0]]) : (tensor<*xf32>, tensor<*xf32>) -> tensor<*xf32>
// CHECK-NEXT:     tf_executor.yield %{{[0-9]*}} : tensor<*xf32>
// CHECK-NEXT:   }
// CHECK-NEXT:   tf_executor.fetch %[[ADD]]#0 : tensor<*xf32>
// CHECK-NEXT: }
// CHECK-NEXT: return %[[GRAPH]] : tensor<*xf32>

// CHECK-LABEL: func @LoopTest() {

func @LoopTest() {
  %0:2 = "_tf.Const"() {device = "", dtype = "tfdtype$DT_INT32", name = "Const", value = dense<1> : tensor<i32>} : () -> (tensor<i32>, !_tf.control)
  %1:2 = "_tf.Enter"(%0#0) {T = "tfdtype$DT_INT32", device = "", frame_name = "while/while_context", is_constant = false, name = "while/Enter", parallel_iterations = 10 : i64} : (tensor<i32>) -> (tensor<*xi32>, !_tf.control)
  %2 = "_tf.NoOp"() {device = "", name = "cluster/pivot"} : () -> !_tf.control
  %3:2 = "_tf.NextIteration.source"() {T = "tfdtype$DT_INT32", device = "", id = 0 : i64, name = "while/NextIteration"} : () -> (tensor<*xi32>, !_tf.control)
  %4:3 = "_tf.Merge"(%3#0, %1#0) {N = 2 : i64, T = "tfdtype$DT_INT32", device = "", name = "while/Merge"} : (tensor<*xi32>, tensor<*xi32>) -> (tensor<*xi32>, tensor<i32>, !_tf.control)
  %5:2 = "_tf.Const"(%4#2) {device = "", dtype = "tfdtype$DT_INT32", name = "while/Less/y", value = dense<2> : tensor<i32>} : (!_tf.control) -> (tensor<i32>, !_tf.control)
  %6:2 = "_tf.Less"(%4#0, %5#0) {T = "tfdtype$DT_INT32", device = "", name = "while/Less"} : (tensor<*xi32>, tensor<i32>) -> (tensor<*xi1>, !_tf.control)
  %7:2 = "_tf.LoopCond"(%6#0) {device = "", name = "while/LoopCond"} : (tensor<*xi1>) -> (tensor<i1>, !_tf.control)
  %8:3 = "_tf.Switch"(%4#0, %7#0) {T = "tfdtype$DT_INT32", _class = ["loc = @while/Merge"], device = "", name = "while/Switch"} : (tensor<*xi32>, tensor<i1>) -> (tensor<*xi32>, tensor<*xi32>, !_tf.control)
  %9:2 = "_tf.Exit"(%8#0) {T = "tfdtype$DT_INT32", device = "", name = "while/Exit"} : (tensor<*xi32>) -> (tensor<*xi32>, !_tf.control)
  %10:2 = "_tf.Identity"(%8#1) {T = "tfdtype$DT_INT32", device = "", name = "while/Identity"} : (tensor<*xi32>) -> (tensor<*xi32>, !_tf.control)
  %11:2 = "_tf.Const"(%10#1) {device = "", dtype = "tfdtype$DT_INT32", name = "while/Add/y", value = dense<3> : tensor<i32>} : (!_tf.control) -> (tensor<i32>, !_tf.control)
  %12:2 = "_tf.Add"(%10#0, %11#0) {T = "tfdtype$DT_INT32", device = "", name = "while/Add"} : (tensor<*xi32>, tensor<i32>) -> (tensor<*xi32>, !_tf.control)
  %13 = "_tf.ControlTrigger"(%2, %12#1, %9#1) {_tpu_replicate = "cluster", device = "", name = "gradients/while/mul_2_Da30D05wlPU_grad/SymbolicGradient/b_sync"} : (!_tf.control, !_tf.control, !_tf.control) -> !_tf.control
  %14 = "_tf.NextIteration.sink"(%12#0, %13) {T = "tfdtype$DT_INT32", device = "", id = 0 : i64, name = "while/NextIteration"} : (tensor<*xi32>, !_tf.control) -> (!_tf.control)
  return
}

// CHECK-NEXT:   tf_executor.graph {
// CHECK-NEXT:     %[[CONST:[0-9]*]]:2 = tf_executor.island {
// CHECK-NEXT:       %{{[a-z0-9]*}} = "tf.Const"() {device = "", dtype = "tfdtype$DT_INT32", name = "Const", value = dense<1> : tensor<i32>} : () -> tensor<i32>
// CHECK-NEXT:       tf_executor.yield %{{[a-z0-9]*}} : tensor<i32>
// CHECK-NEXT:     }
// CHECK-NEXT:     %[[ENTER:[0-9]*]]:2 = tf_executor.Enter %[[CONST]]#0 frame "while/while_context" : (tensor<i32>) -> (tensor<*xi32>, !tf_executor.control) {T =  "tfdtype$DT_INT32", device =  "", name =  "while/Enter"}
// CHECK-NEXT:     %[[NOOP:[0-9]*]] = tf_executor.island {
// CHECK-NEXT:       "tf.NoOp"() {device = "", name = "cluster/pivot"} : () -> ()
// CHECK-NEXT:       tf_executor.yield
// CHECK-NEXT:     }
// CHECK-NEXT:     %[[NEXTIT_SRC:[0-9]*]]:3 = tf_executor.NextIteration.Source : tensor<*xi32> {T =  "tfdtype$DT_INT32", device =  "", id =  0 : i64, name =  "while/NextIteration"}
// CHECK-NEXT:     %[[MERGE:[0-9]*]]:3 = tf_executor.Merge %[[NEXTIT_SRC]]#0, %[[ENTER]]#0 : tensor<*xi32> {N = 2 : i64, T =  "tfdtype$DT_INT32", device =  "", name =  "while/Merge"}
// CHECK-NEXT:     %[[CONST_LESS:[0-9]*]]:2 = tf_executor.island(%[[MERGE]]#2) {
// CHECK-NEXT:       %{{[a-z0-9]*}} = "tf.Const"() {device =  "", dtype =  "tfdtype$DT_INT32", name =  "while/Less/y", value =  dense<2> : tensor<i32>} : () -> tensor<i32>
// CHECK-NEXT:       tf_executor.yield %{{[a-z0-9]*}} : tensor<i32>
// CHECK-NEXT:     }
// CHECK-NEXT:     %[[LESS:[0-9]*]]:2 = tf_executor.island {
// CHECK-NEXT:       %{{[a-z0-9]*}} = "tf.Less"(%[[MERGE]]#0, %[[CONST_LESS]]#0) {T =  "tfdtype$DT_INT32", device =  "", name =  "while/Less"} : (tensor<*xi32>, tensor<i32>) -> tensor<*xi1>
// CHECK-NEXT:       tf_executor.yield %{{[a-z0-9]*}} : tensor<*xi1>
// CHECK-NEXT:     }
// CHECK-NEXT:     %[[COND:[0-9]*]]:2 = tf_executor.LoopCond %[[LESS:[0-9]*]]#0 : (tensor<*xi1>) -> (tensor<i1>, !tf_executor.control) {device =  "", name =  "while/LoopCond"}
// CHECK-NEXT:     %[[SWITCH:[0-9]*]]:3 = tf_executor.Switch %[[MERGE]]#0, %[[COND]]#0 : tensor<*xi32> {T =  "tfdtype$DT_INT32", _class =  ["loc = @while/Merge"], device =  "", name =  "while/Switch"}
// CHECK-NEXT:     %[[EXIT:[0-9]*]]:2 = tf_executor.Exit %[[SWITCH]]#0 : tensor<*xi32> {T =  "tfdtype$DT_INT32", device =  "", name =  "while/Exit"}
// CHECK-NEXT:     %[[IDENTITY:[0-9]*]]:2 = tf_executor.island {
// CHECK-NEXT:       %{{[a-z0-9]*}} = "tf.Identity"(%[[SWITCH]]#1) {T =  "tfdtype$DT_INT32", device =  "", name =  "while/Identity"} : (tensor<*xi32>) -> tensor<*xi32>
// CHECK-NEXT:       tf_executor.yield %{{[a-z0-9]*}} : tensor<*xi32>
// CHECK-NEXT:     }
// CHECK-NEXT:     %[[CONST_ADD:[0-9]*]]:2 = tf_executor.island(%[[IDENTITY]]#1) {
// CHECK-NEXT:       %{{[a-z0-9]*}} = "tf.Const"() {device =  "", dtype =  "tfdtype$DT_INT32", name =  "while/Add/y", value = dense<3> : tensor<i32>} : () -> tensor<i32>
// CHECK-NEXT:       tf_executor.yield %{{[a-z0-9]*}} : tensor<i32>
// CHECK-NEXT:     }
// CHECK-NEXT:     %[[ADD:[0-9]*]]:2 = tf_executor.island {
// CHECK-NEXT:       %{{[0-9]*}} = "tf.Add"(%[[IDENTITY]]#0, %[[CONST_ADD]]#0) {T =  "tfdtype$DT_INT32", device =  "", name =  "while/Add"} : (tensor<*xi32>, tensor<i32>) -> tensor<*xi32>
// CHECK-NEXT:       tf_executor.yield %{{[0-9]*}} : tensor<*xi32>
// CHECK-NEXT:     }
// CHECK-NEXT:     %[[CT:[0-9]*]] = tf_executor.ControlTrigger %[[NOOP]], %[[ADD]]#1, %[[EXIT]]#1 {_tpu_replicate = "cluster", device = "", name = "gradients/while/mul_2_Da30D05wlPU_grad/SymbolicGradient/b_sync"}
// CHECK-NEXT:     tf_executor.NextIteration.Sink [%[[NEXTIT_SRC]]#1] %[[ADD]]#0, %[[CT]] : tensor<*xi32> {T =  "tfdtype$DT_INT32", device =  "", id = 0 : i64, name =  "while/NextIteration"}
// CHECK-NEXT:     tf_executor.fetch
// CHECK-NEXT:   }
// CHECK-NEXT:   return
