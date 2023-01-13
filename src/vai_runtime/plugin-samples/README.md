# Plugins

## Overview

[XIR]() based toolchain was released in Vitis AI 1.1 for the first time as a unified compilation-deployment process from edge to cloud. Furthermore, we introduced Plugin in Vitis AI 1.3 to enable Vitis AI users to accelerate DPU *unsupported* operations with their own RTL/HLS IPs or CPU function in standard Vitis AI flow. 

## Introduction

Vitis AI compiler works in the context of a framework independent XIR graph generated from deep learning frameworks, the Parser removes all the framework-specific attributes in CNN models and transforms models into XIR-based computing graphs. The DPU Compiler will divide the computing graph into different subgraphs, leverages heterogeneous optimizations and generates corresponding optimized machine codes for subgraphs. Plugin is the compiler for customized accelerator which does exact what DPU compiler does to the graph. Pick the operators your accelerator can support, create subgraphs for them and attach whatever the runtime needs to drive the accelerator to the subgraph. The plugins will be executed sequentially before DPU compilation and the normal DPU compiler will only work on the rest part of the graph. 

## How to Implement a Plugin

- Create your plugin inherited from the interface class Plugin

    ```cpp
    // Interface class Plugin
    class Plugin {
    public:
    // Return plugin name
    virtual std::string get_plugin_name() = 0;
    
    // Return device name
    // This will be set to the subgraphs created by Plugin as attribute "device"
    virtual std::string get_device() = 0;
    
    // Return runner library name
    // the key could be 'ref', 'sim' or 'run' and value is the corresponding library name
    // 'ref' for the CPU reference implementation library
    // 'sim' for the simulator implementation library
    // 'run' for the runtime implementation libray
    // This will be set to the subgraphs created by Plugin as attribute "runner"
    virtual std::map<std::string, std::string> get_runner() = 0;
    
    // Pick the operators you needs, and merge them into one subgraph or several subgraphs
    // according to your requirements and the topological structure.
    // `PluginHelper` provides some helper functions to make this easier to implement 
    virtual std::set<xir::Subgraph*> partition(xir::Graph* graph) = 0;
    
    // Compile the subgraph and attach everything the runner needs on the subgraph.
    // This function will be applied to each subgraph created by partition() function.
    virtual void compile(xir::Subgraph* subgraph) = 0;
    };
    ```

- Implement those member functions.  
  - In the beginning of the compilation, the graph will be initialized with a root subgraph which contains all the operators and each operator will be wrapped by a child subgraph of root subgraph. 
  - In `partition` function, you need find all the operators your accelerator can support and then you can merge them into larger subgraphs. 
  - A runner will be created for each subgraph later when deploying
  - Whether should you merge the child subgraphs into a larger one or which child subgraphs should you merge depends the IP capability and the topological structure(data dependencies). Sometimes you cannot merge some operators because if you do so there will be a ring in dependency graph of the subgraphs, which means it is not deployable.
  - To make it easier to implement, `PluginHelper` provides some useful helper functions

    ```cpp
    class PluginHelper {
    public:
    // Return unassigned child subgraph which contains op with specific name
    static xir::Subgraph* filter_by_name(xir::Graph* graph,
                                        const std::string& name);
    
    // Return unassigned child subgraphs which contain ops with specific type
    static std::set<xir::Subgraph*> filter_by_type(xir::Graph* graph,
                                                    const std::string& type);
    
    // Return unassigned child subgraphs whose ops construct same structure
    // as the template using subgraph isomorphism algorithm
    static std::set<xir::Subgraph*> filter_by_template(
        xir::Graph* graph, xir::GraphTemplate* temp);
    
    // Return unassigned child subgraphs filterd by customized function,
    // this function helps you find all the unassigned subgraphs and call the
    // customized function with them
    static std::set<xir::Subgraph*> filter(
        xir::Graph* graph,
        std::function<std::set<xir::Subgraph*>(std::set<xir::Subgraph*>)> func);
    
    // Try to merge the subgraphs into one as far as possible. It may
    // return more than one subgraph depending on the data dependencies
    static std::set<xir::Subgraph*> merge_subgraph(std::set<xir::Subgraph*> subgraph_list);
    };
    ```

- Implement `compile` function
  -  This function will be called for each subgraph returned by `partition` function. The implementation totally depends on your IP and runtime, so do whatever you want and you can attach information on the subgraph. 


## Build Plugin

In your plugin implementation `*.cpp` files, you need to create an `extern get_plugin()` function which compiler will use to create a plugin instance and build the implementations into a shared library.

```cpp 
extern "C" plugin* get_plugin() { return new YOURPLUGIN(); }
```

## Use Plugin in Vitis-AI Compiler

The `plugins` option in command line can be used to pass your plugin library to compiler. For example, if the compiled shared library is `libplugin-demo.so`,  `--options '{"plugins": "plugin-demo"}'` should be added to let the compiler execute your plugin. 

During executing your plugin, compiler will `dlopen` the library and make an instance of your plugin by loading your extern function named `get_plugin`. The plugins would be executed sequentially in the order defined by plugin option. 

Example Command:
```sh
vai_c_tensorflow -a /opt/vitis_ai/compiler/arch/DPUCAHX8H/U50/arch.json -f quantize_eval_model.pb -n resnet_v1_50 --options '{"plugins": "plugin-demo"}'
```

For better understand about how compiler works with the plugins specified in command line, some crucial steps are listed below.

- Firstly, he compiler will load the plugin library by dlopen, and find the get_plugin symbol to get the plugin function.

```cpp
Plugin* load_plugin_lib(const std::string& lib) {
  auto dir = "lib" + lib + ".so";
  typedef Plugin* (*PLUGIN)();
  // load the plugin's shared library
  auto handle = dlopen(dir.c_str(), RTLD_LAZY);
  PLUGIN plugin_func = nullptr;
  // touch the get_plugin symbol in shared library,
  // and transform the get_plugin into a c++ function.
  plugin_func = (PLUGIN)dlsym(handle, "get_plugin");
  return plugin_func();
}
```

- Secondly, it will call the `partition` function to get all the subgraphs targeting the customized device.

```cpp
auto subgraphs = plugin->partition(graph);
```

- Then it calls the `compile` function on each subgraph from last step, and set the runner and device information in these subgraphs as attributes.

```cpp
std::for_each(subgraphs .begin(), subgraphs .end(), [&plugin](auto subgraph) {
  plugin->compile(subgraph);
  subgraph->set_attr("runner", plugin->get_runner());
  subgraph->set_attr("device", std::string(plugin->get_device()));
});
```

- Lastly, the compiler will compile the remained subgraphs targeting DPU or CPU.

## Examples

- [SoftMax Plugin for Edge Devices (IP)](samples/plugin-smfc/README.md)
- [SoftMax Plugin for Cloud Devices (CPU)](samples/plugin-smfc-cpu/README.md)
- [Integrating HW Plugins with DPU](../../../../dsa/WAA-TRD/README.md)