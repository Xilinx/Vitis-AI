# Customized XIR OP example

## introduction

In this example, we implement an XIR OP `add`, it simply adds two
input tensors and assuming the two tensors have the same shape.

Regarding to how to register a new XIR OP, please refer to XIR user
manual or Xcompiler user manual. (TODO: add URL links). Here we assume
that `add` OP is already registered.

To implement an XIR OP, we need to follow the below steps.

1. Write a C++ class.
2. Write the constructor function.
3. Write the `calculate` function.
4. Regester the implementation with the macro
4. Build a shared library.
5. deploy it.


## Write `add` OP implementation.

First of all, in `my_add_op.cpp`, we create a C++ class, we may name
the source file or class name whatever we like.

``` c++
// in my_add_op.cpp
class MyAddOp {
};
```

Then, we write the constructor function as below

``` c++
#include <vart/op_imp.h>

class MyAddOp {
    MyAddOp(const xir::Op* op1, xir::Attrs* attrs) : op{op1} {
        // op and attrs is not in use.
    }
public:
   const xir::Op * const op;
};

```

Note that `MyAddOp` must have a public member variable named
`op`. `op` is initialized with the first input argument of the
constructor function, e.g. `op1`. This is the requirement of
`DEF_XIR_OP_IMP`

Then, we write the member function `calculate` function as below.

``` c++
class MyAddOp {
   ...
  int calculate(vart::simple_tensor_buffer_t<float> output,
                std::vector<vart::simple_tensor_buffer_t<float>> inputs) {
    for (auto i = 0u; i < output.mem_size / sizeof(float); ++i) {
      output.data[i] = 0.0f;
      for (auto input : inputs) {
        output.data[i] = output.data[i] + input.data[i];
      }
    }
    return 0;
  }
  ...
}
```

To compile the source file

``` shell
% g++ -fPIC -std=c++17 -c -o  /tmp/my_add_op.o -Wall -Werror -I ~/.local/Ubuntu.18.04.x86_64.Debug/include/ my_add_op.cpp
```

Note that we must use C++17 or above. To build a shared library we
also enable `-fPIC`. We assume `vitis-ai-libary` is installed at
`~/.local/Ubuntu.18.04.x86_64.Debug`


To link to a shared library

``` shell
% mkdir -p /tmp/lib;
% g++ -Wl,--no-undefined -shared -o /tmp/lib/libvart_op_imp_add.so /tmp/my_add_op.o -L ~/.local/Ubuntu.18.04.x86_64.Debug/lib -lglog -lvitis_ai_library-runner_helper -lvart-runner -lxir
```

To test the op implemenation, we need to create a sample XIR graph first as below

```
% ipython;
import xir
g = xir.Graph("simple_graph")
a = g.create_op("a", "data", {"shape": [1,2,2,4], "data_type": "FLOAT32"});
b = g.create_op("b", "data", {"shape": [1,2,2,4], "data_type": "FLOAT32"});
add = g.create_op("add_op", "add",  {"shape": [1,2,2,4], "data_type": "FLOAT32"}, {"input": [a,b]})
root = g.get_root_subgraph()
root.create_child_subgraph()
user_subgraph = root.merge_children(set([g.get_leaf_subgraph(a), g.get_leaf_subgraph(b)]))
cpu_subgraph = root.merge_children(set([g.get_leaf_subgraph(add)]))
user_subgraph.set_attr("device", "USER")
cpu_subgraph.set_attr("device", "CPU")
g.serialize("/tmp/add.xmodel")
```

Instead of above tedious python codes, a xmodel is usually created by
Xcompiler, please refer to Xcompiler user manual. (TODO: add URL
links);


Now we create a sample input files

``` shell
% cd /tmp
% mkdir -p ref
% ipython
import numpy as np
a = np.arange(1, 17, dtype=np.float32)
b = np.arange(1, 17, dtype=np.float32)
a.tofile("ref/a.bin")
b.tofile("ref/b.bin")
c = a + b
c.tofile("ref/c.bin")
```

``` shell
% cd /tmp
% mkdir -p /tmp/dump
% env LD_LIBRARY_PATH=$HOME/.local/Ubuntu.18.04.x86_64.Debug/lib:/tmp/lib $HOME/.local/Ubuntu.18.04.x86_64.Debug/share/vitis_ai_library/test/cpu_task/test_op_imp --graph /tmp/add.xmodel --op "add_op"
```

Note that we add `/tmp/lib` into the search path `LD_LIBRARY_PATH` so
that the CPU runner can find the shared library we just wrote.

The name of shared library is also important and it must be
`libvart_op_imp_<YOUR_OP_TYPE>.so`, the CPU runner use this naming
scheme to find the customized xir::Op implementation.


We can verify that the op is implemented properly by comparing it with the reference result.

```shell
% diff -u <(xxd ref/c.bin) <(xxd dump/add_op.bin)
% xxd ref/c.bin
% xxd dump/add_op.bin
```

## write customized XIR::Op implementation in Python

It is also possible to write XIR::Op implementation in Python.

First all, we create a python package and a module file.

``` shell
% mkdir -p /tmp/demo_add_op/
% cd /tmp/demo_add_op/
% mkdir vart_op_imp/
% touch vart_op_imp/__init__.py
% touch vart_op_imp/add.py
```

Now we update `add.py` as below

``` python
import numpy as np


class add:
    def __init__(self, op):
        pass

    def calculate(self, output, input):
        np_output = np.array(output, copy=False)
        L = len(input)
        if L == 0:
            return
        np_input_0 = np.array(input[0], copy=False)
        np.copyto(np_output, np_input_0)
        for i in range(1, L):
            np_output = np.add(np_output,
                               np.array(input[i], copy=False),
                               out=np_output)

```

or we can have a simplified version as below

```
import numpy as np


class add:
    def __init__(self, op):
        pass

    def calculate(self, output, input):
        np_output = np.array(output, copy=False)
        L = len(input)
        assert L == 2
        np_input_0 = np.array(input[0], copy=False)
        np_input_1 = np.array(input[1], copy=False)
        np_output = np.add(np_input_0, np_input_1, out=np_output)
```

We install the op as below

``` shell
% mkdir -p lib
% ln -sf ~/.local/Ubuntu.18.04.x86_64.Debug/lib/libvart_op_imp_python-cpu-op.so lib/libvart_op_imp_add.so
% ls -l lib
% mkdir -p dump
% env LD_LIBRARY_PATH=$HOME/.local/Ubuntu.18.04.x86_64.Debug/lib:$PWD/lib $HOME/.local/Ubuntu.18.04.x86_64.Debug/share/vitis_ai_library/test/cpu_task/test_op_imp --graph /tmp/add.xmodel --op "add_op"
```

Similiar to the C++ interface, the python module must have a class
whose name is as same as the xir::Op's type. It is `add` in the
example. The class must have a constructor with single argument `op`
in addition to `self`. It is an XIR::Op, please refer XIR Python API
for more detail. Similarly, the class `add` must have a member
function `calculate`, in addition to `self` argument, the first
argument must be named `output`, and following argument names must
comply with the XIR::OpDef associated with the XIR::Op, please refer
to XIR API for more detail.

Note that we create a symbolic link to
`libvart_op_imp_python-cpu-op.so` with name
`libvart_op_imp_add.so`. `libvart_op_imp_python-cpu-op.so` is a bridge
between Python and C++. From C++ side, it searches for
`libvart_op_imp_add.so` as usual, it happened to be the
`libvart_op_imp_python-cpu-op.so`. In
`libvart_op_imp_python_cpu_op.so`, python module name
`vart_op_imp.add` is imported and python search for the module as
usual.

## Conclusion

We demostrate how to implement a customized Op in this article, please refer to the manual for more detail(TODO: add links)
