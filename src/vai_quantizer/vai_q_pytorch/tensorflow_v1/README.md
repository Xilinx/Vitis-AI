<h1 align="center">Prune Model with Vitis Optimizer TensorFlow1</h1>

## Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Pruning](#pruning)
- [APIs](#apis)
- [Examples](#examples)

## Overview

Users have to instantiate a tensorflow session which contains graph and initialized variables (initialized by tensorflow initializers, checkpoint, SavedModel etc.) before pruning. Vitis Optimizer TensorFlow1 will prune the graph **in-place** and provides a method to export frozen pruned graph.

The pruned graph in memory is a sparse graph, which means the shapes of weights are the same as the original ones and the channels to be removed will be set to zeros. The exported frozen pruned graph is a slim graph with smaller size, which means the unnecessary channels will be actually removed

## Installation

Install with pip

```bash
cd tensorflow_v1 && pip install .
```

## Pruning

In short, this method includes two stages: model analysis and pruned model generation. The elaborate prune process is as follows.
### Creating a Baseline Model

Here, a simple MNIST convnet is used.

```python
model = keras.Sequential([
      layers.InputLayer(input_shape=input_shape),
      layers.Conv2D(16, kernel_size=(3, 3), activation="relu"),
      layers.BatchNormalization(),
      layers.MaxPooling2D(pool_size=(2, 2)),
      layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
      layers.MaxPooling2D(pool_size=(2, 2)),
      layers.Flatten(),
      layers.Dropout(0.5),
      layers.Dense(num_classes),
  ])
```
### Instantiating a tensorflow Session

```python
with tf.Session() as sess:
  model, input_shape = mnist_convnet()
  sess.run(tf.global_variables_initializer()) # initializing variables
```

### Creating a Pruning Runner

To get a pruning runner, users have to provide an instance of tensorflow Session, input specs and output node names as follows:
```python
from tf1_nndct.optimization.pruning import IterativePruningRunner

with tf.Session() as sess:
  model, input_shape = mnist_convnet()
  sess.run(tf.global_variables_initializer())
  input_specs={'input_1:0': tf.TensorSpec(shape=(1, 28, 28, 1), dtype=tf.dtypes.float32)}
  pruner = IterativePruningRunner("mnist", sess, input_specs, ["dense/BiasAdd"])
```

### Pruning the Baseline Model
To prune a model, follow these steps:

1. Define a function of type `Callable[[tf.compat.v1.GraphDef], float]` to evaluate model performance. The only argument is a frozen graph which is an intermediate result of pruning process. The pruning runner will perform multiple prunings to find the optimal pruning strategy.

    ```python
    def eval_fn(frozen_graph_def: tf.compat.v1.GraphDef) -> float:
      with tf.compat.v1.Session().as_default() as sess:
        tf.import_graph_def(frozen_graph_def, name="")

        # do evaluation here

        return 0.5 # Returning a constant is for demonstration purpose
    ```
2. Use this evaluation function to run model analysis: 
  Users could specify devices to use for model analysis. The default value is `['/GPU:0']`. If multiple devices are given, the pruning runner will run model analysis on each device in parallel.

    ```python
    pruner.ana(eval_fn, gpu_ids=['/GPU:0', '/GPU:1'])
    ```

3. Determine a pruning sparsity. The ratio indicates the reduction in the amount of floating-point computation of the model in forward pass. `pruned_model's FLOPs = (1 – sparsity) * original_model's FLOPs`. The value of ratio should be in (0, 1):

    ```python
    shape_tensors, masks = pruner.prune(sparsity=0.5)
    ```
    **Note:** `sparsity` is only an approximate target value and the actual pruning ratio may not be exactly equal to this value.<br/>
    *shape_tensors* is a string to NodeDef mapping. The keys are the names of node_defs in graph_def which need to be updated to get a slim graph. *masks* corresponds to variables. It only contains 0s and 1s. After calling this method, the graph within session is pruned and becomes a sparse graph.
    <br/>

4. Export frozen slim graph. Use *shape_tensors* and *masks* returned by method *prune* to generate a frozen slim graph as follows:

    ```python
    slim_graph_def = pruner.get_slim_graph_def(shape_tensors, masks)
    ```

### Fine-tuning a Sparse Model

The graph is pruned **in-place**, so users can fine-tune the pruned graph just like the original one. Tensorflow1 optimizer will apply masks automatically.


## APIs

the pruning runner has three important APIs: *ana*, *prune* and *get_slim_graph_def*.

```python
class IterativePruningRunner(
      self, 
      model_name: str, 
      sess: SessionInterface, 
      input_specs: Mapping[str, tf.TensorSpec], 
      output_node_names: List[str], 
      excludes: List[str]=[])
```

Arguments:
- **model_name**: The name of baseline model
- **sess**: Instance of tensorflow session, containing graph and initialized variables
- **input_specs**: The keys of this mapping is input node names of baseline model
- **output_node_names**: Target output node names
- **exludes**: The names of nodes that skip pruning

Returns:
- Instance of IterativePruningRunner


```python
def ana(self, 
        eval_fn: Callable[[tf.compat.v1.GraphDef], float], 
        gpu_ids: List[str]=['/GPU:0'], 
        checkpoint_interval: int = 10) -> None:
```
Arguments:
- **eval_fn**: The function to evaluate intermediate results of pruning process. Needs to return a float
- **gpu_ids**: A list of strings, indicating devices to run evaluations
- **checkpoint_interval**: This method implements cache mechanism and saves results every *checkpoint_interval* evaluations

Returns:
- None

```python
def prune(
    self,
    sparsity: float=None, 
    threshold: float=None, 
    max_attemp: int=10) -> Tuple[Mapping[str, TensorProto], Mapping[str, np.ndarray]]:
```
There are two modes of pruning: FLOPs based and accuracy based, corresponding to argument *sparsity* and *threshold*. These two arguments MUST NOT be None at the same time. Argument *sparsity* is more prioritised than *threshold*.

Arguments:
- **sparsity**: The ratio indicates the reduction in the amount of floating-point computation of the model in forward pass
- **threshold**: Within range [0, 1]. Indicating the max acceptable relative difference in accuracy between pruned graph and original graph
- **max_attemp**: Pruning runner will find the optimal pruning strategy iteratively and will return after *max_attemp* steps anyway.

Returns:
- **shape_tensors**: A string to NodeDef mapping. The keys are the names of node_defs in graph_def which need to be updated to get a slim graph. The values are target node_def contents
- **masks**: A string to ndarray mapping, correspongding to variables.

```python
def get_slim_graph_def(
  self, 
  shape_tensors: Mapping[str, TensorProto]=None, 
  masks: Mapping[str, np.ndarray]=None) -> tf.compat.v1.GraphDef:
```

The arguments are outputs of method *prune*. Returns a frozen slim graph_def.

## Examples
Please find examples [here](/example/pruning/tensorflow_v1)
