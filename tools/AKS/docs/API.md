Integrating AI Kernel Scheduler in Your Application
===

Users can create their own application pipelines by defining custom graphs and use AI Kernel Scheduler in their applications through the C++ or Python APIs. Please refer to the examples provided in [examples](../examples) directory.

Following are the steps to build applications with AKS C++/Python API:

### Include AKS Header Files / Import the Python Module

AKS C++ headers are installed in the docker container for cloud platforms. For edge platforms, the AKS installer (`.rpm`) is provided to install AKS library.

`AksSystemManagerExt.h` is the only mandatory header to be included in your application.

Other headers in the directory are optional and its usage will be discussed in below sections.

```cpp
// C++
#include "aks/AksSystemManagerExt.h"
```

AKS Python library is in [libs](../libs) directory. Please make sure to add this path to the `PYTHONPATH` environment variable.

```python
# Python
import aks
```

### Get the System Manager Handle

**SysManagerExt** is a global manager which handles all the kernels, graphs and their associated jobs.

```cpp
// C++
auto sysMan = AKS::SysManagerExt::getGlobal();
```
```python
# Python
sysMan = aks.SysManager()
```

### Load Kernels

AKS loads all the kernels using `loadKernels()` method. Provide the directory path containing AKS Kernel Configs and AKS loads all the kernels in that directory. All the kernel configs that comes with AKS package can be found in [kernel_zoo](../kernel_zoo).

```cpp
// C++
sysMan->loadKernels("path-to-kernel-zoo-dir");
```
```python
# Python
sysMan.loadKernels("path-to-kernel-zoo-dir")
```


### Load Graph(s)

Use `loadGraphs()` method to load the graph(s) for execution. This method takes either a,
* Path to a graph JSON: AKS loads graph specified (Single graph)

  OR

* Directory containing graph JSONs: AKS loads all the graphs in the specified directory (Multi-graph)

```cpp
// C++
sysMan->loadGraphs("path-to-graph-or-graphzoo-dir");
```
```python
# Python
sysMan.loadGraphs("path-to-graph-or-graphzoo-dir")
```

### Get a Handle to the Graph

Each loaded graph can be identified by an unique name provided in the graph JSON. Provide this graph name to `getGraph()` method to get a handle to the loaded graph. This handle can be used to enqueue jobs to this graph.

```cpp
// C++
auto graph = sysMan->getGraph("graph-name");
```
```python
# Python
graph = sysMan.getGraph("graph-name")
```

### Enqueue Jobs

Jobs are enqueued asynchronously to SysManagerExt using its `enqueueJob()` method.

#### Enqueue Jobs in C++

In C++ API, `enqueueJob()` takes 4 arguments:
1. **Graph Handle:** It specifies to which graph the job is being pushed.
2. **Image Path:** A string representing the image path. Typically, first node in an computer vision pipeline loads an image. If it is not the case, pass an empty string.
3. **Buffers:** (*Optional*) This is a `std::vector<std::unique_ptr<vart::TensorBuffer>>`. This lets the user to directly pass any number of input buffers like decoded video frames. By default, an empty vector is passed.
4. **User Args:** (*Optional*) This is a reference to `AKS::NodeParams` object. This lets user to pass any job-specific information and this information will be accessible to every node in the pipeline. By default, a `nullptr` is passed.

It returns a `C++ future` object which can be used to wait for the results asynchronously.

```cpp
// C++
// create the tensor buffer vector
std::vector<std::unique_ptr<vart::TensorBuffer>> v(3);
auto fut = sysMan->enqueueJob (graph, imagePath, std::move(v), nullptr);
```

:pushpin: **Note:** Include `aks/AksTensorBuffer.h`, `aks/AksBatchTensorBuffer` and `aks/AksNodeParams.h` to get `AKS::AksTensorBuffer`, `AKS::AksBatchTensorBuffer` and `AKS::NodeParams` respectively.


#### Enqueue Jobs in Python

In Python API, `enqueueJobs()` takes only 2 arguments which are same as in C++ API. Custom input buffers or any UserArgs are not supported by Python API. This limits Python applications to have graphs whose first node should always load image from the image path provided, not directly pass any decoded video frame.

Also, Python API doesn't return anything as in C++ API.

```python
# Python
sysMan.enqueueJob(graph, imagePath)
```
:pushpin: **Note:** Not all features of C++ API are exposed to Python API.

### Wait For Results

Since jobs are enqueued asynchronously, the main thread has to wait for jobs to finish to get the output. This is a blocking call.

AKS supports waiting for results at three levels.

1. **System level:** Wait for all the jobs enqueued to SysManager to finish.

    Eg : [examples/resnet50.cpp](../examples/resnet50.cpp) uses this to calculate the accuracy of Resnet50 model on a test dataset.

    ```cpp
    // C++
    sysMan->waitForAllResults();
    ```
    ```python
    # Python
    sysMan.waitForAllResults()
    ```

1. **Graph level:** Provide a graph handle to wait for all the jobs enqueued for that particular graph to finish without interrupting the execution of other graphs.

    Eg : [examples/googlenet_resnet50.cpp](../examples/googlenet_resnet50.cpp) uses this to get the accuracy of both Googlenet and Resnet50 in one shot.

    ```cpp
    // C++
    sysMan->waitForAllResults(graph);
    ```
    ```python
    # Python
    sysMan.waitForAllResults(graph)
    ```

1. **Job level:** : This let's the user to get the output of a particular job without blocking other jobs that are still running. This is done by calling `.get()` method of `std::future` object returned for every `enqueueJob()` call.

    ```cpp
    // C++
    auto fut = sysMan->enqueueJob (graph, imagePath, std::move(v), nullptr);
    std::vector<std::unique_ptr<vart::TensorBuffer>> outDD = fut.get();
    ```

    :pushpin: **Note:** AKS doesn't guarantee any sequential consistency. There is a possibility that second job might finish before the first job. Therefore, when sequentiality is important, it is recommended to use Job-level waiting. For performance consideration, waiting may have to be done in a separate thread. 

    :pushpin: **Note:** Job level waiting is not exposed with Python API.

### Report Results (Optional)

This is an optional step to report any results. AKS `report()` method invokes `report()` method for every node in the graph, if the kernel used by the node has implemented a `report()` method.

For example, AKS classification accuracy kernel has implemented this method to report out the Top-1 and Top-5 accuracies.

```cpp
// C++
sysMan->report(graph);
```
```python
# Python
sysMan.report(graph)
```

:bulb: **INFO:** More details about the APIs above and their arguments can be found in the headers files `AksSysManagerExt.h` (for C++ interface) & [aks.py](../libs/aks.py) (for Python interface).

## Creating Custom Graphs for AI Kernel Scheduler

AKS graph is a *Directed Acyclic Graph (DAG)* composed of execution nodes written in JSON format. Each node should be an instance of a kernel defined in [kernel_zoo](../kernel_zoo).

:bulb: **INFO:** Please refer to pre-defined graphs in [graph_zoo](../graph_zoo) directory for reference.

A graph has only two fields:

1. **`"graph_name"`**: This is any name to uniquely identify the graph
1. **`"node_list"`**: This is the list of nodes in the graph. Each node has mainly 3 fields.
    1. **`"node_name"`**: Name of the node
    1. **`"node_params"`**: Parameters of the node. It depends upon the kernel definition.
    1. **`"next_node"`**: It is a list of node names to which this node is connected to. This defines the connectivity in the DAG.
        * For the last node, this field must be empty.

:pushpin: Currently, AKS expects the graph to have single input and single output.

## Creating Custom AKS Kernel

:bulb: **INFO:** AKS provides a set of kernels that covers a wide range of use-cases in [kernel_zoo](../kernel_zoo). Please check them out.

Creating a custom kernel requires two components.

1. **Kernel Interface**
   - This is the interface to the kernel that contains properties of a kernel such as kernel description, input parameters, kernel library path etc. AKS uses `.JSON` files for kernel interfaces.

1. **Kernel Implementation**
   - This is the actual code that will be executed for a kernel.

We will go in detail with a reference kernel, `Add Kernel`, which simply adds a constant value to the input.

:bulb: **INFO:** From Vitis-AI v1.3 onwards, AKS provides all the kernel sources. Please refer to them in [kernel_src](../kernel_src) directory.

### Kernel Interface

AKS crawls through all the kernel JSONs in [kernel_zoo](../kernel_zoo) to get information about each kernel present. Please refer [kernel JSON](../kernel_zoo/kernel_add.json) for `Add Kernel`.

Kernel JSON should have following fields:

- **kernel_name** : Unique identifier for the kernel
- **description** : A description about the kernel.
- **kernel_lib** : Path to the shared library of a kernel
- **param_list** : List of parameters that need to be specified in the graph json. Field name and data type should be mentioned. For eg: For int variable named `adder`, use `"adder": { "type":"int" }`. For array type, use it like this : `"mean": { "type":"float_array" }`.
- **kernel_type** : Possible values - *"cpp"*
- **device_type** : Possible values - *"cpu"*, *"fpga"*

There are some optional fields:

- **num_cu** : Number of CPU threads to be used for a kernel. They are shared among multiple nodes of same kernel, if present, in the graph.

### Kernel Implementation

All kernels should inherit from an abstract class, **`KernelBase`** (*Please see the `aks/AksKernelBase.h` for detailed documentation*). Check the [implementation](../kernel_src/add/add_kernel.cpp) of `Add Kernel` for reference.

There should be a generator function named, **`getKernel()`**, which returns a pointer to Kernel instance. This function should be wrapped in `extern "C"`. If kernel generation requires any specific parameters, it can be passed to `getKernel()` through **NodeParams**(See: `aks/AksNodeParams.h`). This is a generic dict structure filled by AKS with the data provided in the graph json.

**`KernelBase::exec_async()`** is the only pure virtual method. It takes four arguments.

1. **`inputs`** :  It is a `vector<vart::TensorBuffer*>`. `vart::TensorBuffer` is a base class for N-dimensional array structure to store input/output of each node. Inputs to a node are automatically filled with previous nodes' outputs by AKS. `vart::TensorBuffer` can be dynamically casted to `AKS::AksTensorBuffer` or `AKS::AksBatchTensorBuffer` based on the requirement.

2. **`outputs`** : It is also a `vector<vart::TensorBuffer*>`. `exec_async()` of custom kernel should create as many outputs it wants and keep them in output.

3. **`NodeParams* params`** : It is populated by AKS with all the node parameters from the graph json. This parameter is unique per node. If graph contains two nodes of same kernel, each of them will have their own `params` so that kernel can identify each node based on its param.

4. **`DynamicParamValues* dynParams`** : It is passed by the user as an argument to `enqueueJob()`. This includes any job specific input params and it is common to all the nodes in a graph. For eg: ground truth file for each input image in an object detection task.

:bulb: **INFO:** All `vart::TensorBuffer` registered as inputs/outputs inside kernels are automatically managed by AKS.

Once the implementation is ready, it should be compiled to a shared library and keep it in [libs](../libs) directory. Please refer the [CMakeLists.txt](../kernel_src/add/CMakeLists.txt) for `Add Kernel`.