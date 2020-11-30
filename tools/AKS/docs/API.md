## Integrating AI Kernel Scheduler in Your Application

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
3. **Buffers:** (*Optional*) This is a `vector<AKS::DataDescriptor>`. This lets the user to directly pass any number of input buffers like decoded video frames. By default, an empty vector is passed.
4. **User Args:** (*Optional*) This is a reference to `AKS::NodeParams` object. This lets user to pass any job-specific information and this information will be accessible to every node in the pipeline. By default, a `nullptr` is passed.

It returns a `C++ future` object which can be used to wait for the results asynchronously.

```cpp
// C++
// create the data descriptor vector
std::vector<AKS::DataDescriptor> v(3);
auto fut = sysMan->enqueueJob (graph, imagePath, std::move(v), nullptr);
```

:pushpin: **Note:** Include `aks/AksDataDescriptor.h` and `aks/AksNodeParams.h` to get `AKS::DataDescriptor` and `AKS::NodeParams` respectively


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

    Eg : [examples/tinyyolov3_video.cpp](../examples/tinyyolov3_video.cpp) uses this feature to get the boxes detected in every video frame sent to tiny-yolo-v3 network and draw boxes on the input image.

    ```cpp
    // C++
    auto fut = sysMan->enqueueJob (graph, imagePath, std::move(v), nullptr);
    vector<AKS::DataDescriptor> outDD = fut.get();
    ```

    :pushpin: **Note:** AKS doesn't guarantee any sequential consistency. There is a possibility that second job might finish before the first job. Therefore, when sequentiality is important, it is recommended to use Job-level waiting. For performance consideration, waiting may have to be done in a separate thread. Please refer to [examples/tinyyolov3_video.cpp](../examples/tinyyolov3_video.cpp) to see how these techniques are used to run object detection on a video input.

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

1. **`inputs`** :  It is a `vector<DataDescriptor*>`. [DataDescriptor](../ext/AksDataDescriptor.h) is a N-dimensional array structure to store input/output of each node. Inputs to a node are automatically filled with previous nodes' outputs by AKS.

2. **`outputs`** : It is also a `vector<DataDescriptor*>`. `exec_async()` of custom kernel should create as many outputs it wants and keep them in output.

3. **`NodeParams* params`** : It is populated by AKS with all the node parameters from the graph json. This parameter is unique per node. If graph contains two nodes of same kernel, each of them will have their own `params` so that kernel can identify each node based on its param.

4. **`DynamicParamValues* dynParams`** : It is passed by the user as an argument to `enqueueJob()`. This includes any job specific input params and it is common to all the nodes in a graph. For eg: ground truth file for each input image in an object detection task.

:bulb: **INFO:** All `DataDescriptors` registered as inputs/outputs inside kernels are automatically managed by AKS.

Once the implementation is ready, it should be compiled to a shared library and keep it in [libs](../libs) directory. Please refer the [CMakeLists.txt](../kernel_src/add/CMakeLists.txt) for `Add Kernel`.


## Creating Python Kernels

AKS kernels can also be written with Python. This enables users exploit Numpy features for fast array processing.

### Invoking a Python Kernel from the Graph JSON

[graph_facedetect.json](../graph_zoo/graph_facedetect.json) uses a Python kernel for post-processing.

Python Kernel in Graph JSON should have following information:

1. Kernel type should be `"Python"`

1. **`module`** : filename of Python module file stored in [libs/pykernels](../libs/pykernels)
    - *Notice that, extension `.py` is removed*
    - For example, if filepath is `libs/pykernels/pyimread.py`, module name should be `pyimread`

1. **`kernel`** :  name of the Python kernel class inside module file
    - Multiple kernels can be kept inside a single module file

1. **`pyargs`** : This is a list of strings in JSON.
    - Each string is written as key-value pair that you would pass to create a typical Python Dict.
    - AKS internally combine all these strings to a single string that looks like a Python Dict creation and stored inside the Node Params
    - Later Python kernel can access it through NodeParams Python API and call `eval` on it to get a Python Dict.

### Writing Python Kernels

Some examples for Python kernels are provided in [libs/pykernels](../libs/pykernels).

[pyimread.py](../libs/pykernels/pyimread.py) is a Python Kernel for image preprocessing and [postproc.py](../libs/pykernels/postproc.py) is a Python Kernel for FDDB face-detection postprocessing.

A Python kernel can be created as follows:

1. Create a Python Module in [libs/pykernels](../libs/pykernels).
   - This is the `.py` file)

1. Import AKS Python module by `import aks`
   - This is for exposing AKS Data Structures to Python

1. Kernel should be a Python class with following methods:

    - **`Constructor`** has a single argument `params` which is equivalent to `NodeParams` in C++.
        1. This contains the various data you passed to the Python node in graph.json. Remember, it is unique to a node in the graph and same for all the jobs.
        2. All the strings that you passed to Python node in graph.json is combined to a single string and can be accessed from params
        3. Once you have the string, directly call `eval` to get a Python dictionary of these arguments.

    - **`exec_async`** takes three arguments
        1. `inputs` : List of input numpy arrays. They all will be `c_contiguous` and `np.float32` type.
        2. `params` : This is same as the `NodeParams` again.
        3. `dynParams` : This contain any arguments sent by the user while enqueuing a job to sysManager. Eg : image path.
        4. The output is always a `list of numpy arrays`
        5. Each numpy array should be `c_contiguous` as well as `np.float32` type

    - **`wait`** is not currently supported. So you can just `pass` it.

    - **`Destructor`** can be user-defined, else `pass` it.

### Limitations of Python Kernel

Current implementation of Python Kernel has some limitations and some of them might impact its overall performance.

1. All Python kernels should be kept at [libs/pykernels](../libs/pykernels) directory
1. Every time a Python kernel is called, input & output buffers are copied between C++ & Python. This might impact performance based on the size of buffers.
1. All inputs to Python kernel are `c_contiguous` and `np.float32` type. And the kernel must ensure that all of its outputs are also `c_contiguous` and `np.float32`. The kernel has to do explicit casting internally if required.
1. Python Kernel is not thread-safe. So only one AKS thread will be used to dispatch Python kernel, even if your graph has multiple Python nodes.
1. Currently, all Python kernels are blocking kernels. Async kernels are not supported.
