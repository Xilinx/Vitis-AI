# Keras Weight Clustering API Design

| Status        | Accepted                                       |
:---------------|:----------------------------------------------|
| **Author(s)** | Mohamed Nour Abouelseoud (mohamednour.abouelseoud@arm.com), Aron Virginas-Tar (aron.virginas-tar@arm.com), Anton Kachatkou (anton.kachatkou@arm.com)|
| **Sponsor**   | Alan Chiao (alanchiao@google.com)             |
| **Updated**   | 2020-05-07                                    |

## Overview

Weight clustering is an optimization technique that facilitates model compression by reducing the number of unique weights in the model. See the related proposal document [here](../proposals/20200205-weight-clustering-proposal.md) for background and motivation for this technique.

From a practical standpoint, the implementation uses a lookup table to hold the cluster centroid values during model training. The weight array is populated with a 'gather' operation so that during back propagation the gradients can be calculated in the normal way. The lookup table is then adjusted using the cumulative gradient values for the weights that correspond to the same centroid.

The number of unique values required, as well as the way the cluster centroids are initialized are passed in as parameters.

The initial values of cluster centroids are fine-tuned during a subsequent training step.

## Design Proposal

### Public API Summary

Clustering is intended to be used similarly to pruning, therefore the interface is intentionally made very similar to the one described in https://www.tensorflow.org/model_optimization/guide/pruning/.

All APIs are exposed in the `tfmot.clustering.keras` package &mdash; e.g. `tfmot.clustering.keras.cluster_weights()`. The following APIs are available for different aspects of clustering:

* `cluster_weights()`

  the main function to be used for clustering a Keras model or a layer. It configures either all supported layers in a model or a specific layer to be clustered;

  ```python
  def cluster_weights(to_cluster,
                      number_of_clusters,
                      cluster_centroids_init,
                      **kwargs):
  """Modify a keras layer or model to be clustered during training.

  This function wraps a keras model or layer with clustering functionality
  which clusters the layer's weights during training. For examples, using
  this with number_of_clusters equals 8 will ensure that each weight tensor has
  no more than 8 unique values.

  Before passing to the clustering API, a model should already be trained and
  show some acceptable performance on the testing/validation sets.

  The function accepts either a single keras layer
  (subclass of `keras.layers.Layer`), list of keras layers or a keras model
  (instance of `keras.models.Model`) and handles them appropriately.

  If it encounters a layer it does not know how to handle, it will throw an
  error. While clustering an entire model, even a single unknown layer would
  lead to an error.

  Arguments:
      to_cluster: A single keras layer, list of keras layers, or a
        `tf.keras.Model` instance.
      number_of_clusters: the number of cluster centroids to form when
        clustering a layer/model. For example, if number_of_clusters=8 then only
        8 unique values will be used in each weight array.
      cluster_centroids_init: how to initialize the cluster centroids.
        Expects the name of a class that implements the
        `AbstractCentroidsInitialisation` interface. The following
        implementations are provided within the API:
          1. `RandomCentroidsInitialisation` : centroids are sampled using the
          uniform distribution between the minimum and maximum weight values in
          a given layer
          2. `DensityBasedCentroidInitialisation` : density-based sampling.
          First, cumulative distribution function is built for weights, then
          y-axis is evenly spaced into number_of_clusters regions. After this
          the corresponding x values are obtained and used to initialize
          cluster centroids.
          3. `LinearCentroidsInitialisation` : cluster centroids are evenly
          spaced between the minimum and maximum values of a given weight.
      **kwargs: Additional keyword arguments to be passed to the keras layer.
        Ignored when to_cluster is not a keras layer.

  Returns:
    Layer or model modified to include clustering related metadata.

  Raises:
    ValueError: if the keras layer is unsupported, or the keras model contains
    an unsupported layer.
  """
  ```

* `cluster_scope()`

  provides a scope for deserializing clustered layers and models;

  ```python
  def cluster_scope():
  """Provides a scope in which Clustered layers and models can be deserialized.

  If a keras model or layer has been clustered, it needs to be within this scope to be successfully deserialized.

  Returns:
      Object of type `CustomObjectScope` with clustering objects included.
  """
  ```

* `strip_clustering()`

  ```python
  def strip_clustering(model):
  """Strip clustering wrappers from the model.

  Once a model has been clustered, this method can be used
  to restore the original model with the clustered weights.

  Only sequential and functional models are supported for now.

  Arguments:
      model: A `tf.keras.Model` instance with clustered layers.

  Returns:
    A keras model with clustering wrappers removed.

  Raises:
    ValueError: if the model is not a `tf.keras.Model` instance.
    NotImplementedError: if the model is a subclass model.

  The exported_model and the orig_model have the same structure.
  """
  ```

  strips the clustering-specific information from a clustered model and restores the original model structure with the clustered weights.

For further details about the API and how to use these functions, please see the tutorials and examples below.

### Tutorials & Examples

In general, the clustering workflow consists of several distinct steps, as follows:

1. **Clustering** &mdash; run clustering on a trained model by invoking the `cluster_weights()` function with the desired clustering parameters:

    ```python
    clustered_model = cluster.cluster_weights(model, **clustering_params)
    ```

2. **Fine-tuning** &mdash; train the clustered model to fine-tune the initial values of cluster centroids:

   ```python
   clustered_model.fit(x_training_data, y_training_data, epochs=1)
   ```

3. **Stripping** &mdash; call `strip_clustering()` to strip the clustering-specific information from the model and re-obtain the original model structure:

   ```python
   stripped_model = cluster.strip_clustering(clustered_model)
   ```

The code snippets below illustrate common usage patterns of the clustering API. For the sake of brevity, the clustering examples presented here only deal with the 1st step of the above workflow.

A more comprehensive end-to-end example on how to use the API to cluster and save a model is provided in the examples directory, under *clustering*.

#### Cluster an Entire Model

When passing an entire model to `cluster_weights()`, the tool will determine which layers are clusterable and will cluster all of them.

```python
clustering_params = {
    'number_of_clusters': 8,
    'cluster_centroids_init': clustering_centroids.LinearCentroidsInitialisation
}

# Clustering a sequential model
model = keras.Sequential([
    layers.Dense(10, input_shape=(10,)),
    layers.Dense(10),
])

# Train model here
# ...

clustered_model = cluster.cluster_weights(model, **clustering_params)
```

```python
# Clustering a functional model
i1 = keras.Input(shape=(10,))
i2 = keras.Input(shape=(10,))
x1 = layers.Dense(10)(i1)
x2 = layers.Dense(10)(i2)
outputs = layers.Add()([x1, x2])

model = keras.Model(inputs=[i1, i2], outputs=outputs)

# Train model here
# ...

clustered_model = cluster.cluster_weights(model, **clustering_params)
```

#### Cluster Some Layers

Individual layers within a model can be clustered selectively by invoking `cluster_weights()` on a layer object, as follows:

```python
i1 = keras.Input(shape=(10,))
i2 = keras.Input(shape=(10,))
x1 = cluster.cluster_weights(keras.layers.Dense(10), **clustering_params)(i1)
x2 = keras.layers.Dense(10)(i2)
outputs = keras.layers.Add()([x1, x2])

clustered_model = keras.Model(inputs=[i1, i2], outputs=outputs)
```
In the above example model the weights of the 1st `Dense` layer will be clustered, whereas the weights of the 2nd layer will not be clustered.

NOTE: Since clustering can only be applied to a pre-trained model, in real-life selective clustering use cases, weights will have to be loaded into the model at creation, requiring the additional use of the `tf.keras.models.clone_model` API.

#### Export Clustered Model

```python
clustered_model = cluster.cluster_weights(model, **clustering_params)
stripped_model = cluster.strip_clustering(clustered_model)
keras.models.save_model(stripped_model, keras_file)
```

#### Import Clustered Model

A clustered Keras model needs to be loaded within a `cluster_scope` in order to be successfully deserialized:

```python
with cluster_scope():
  loaded_model = keras.models.load_model(keras_file)
```

## Compatibilty

### At Launch

Users can apply weight clustering with the following APIs:

* Model building: `tf.keras` with only Sequential and Functional models
* TensorFlow versions: TF 1.x for versions 1.14+ and 2.x.
* TensorFlow execution mode: both graph and eager

### Roadmap

This is a list of possible future improvements and features that are currently under consideration:

* Support for custom clusterable layers &mdash; implemented already to some extent, but the API needs further refinement.
* Sparsity-aware clustering, to ensure that the sparsity of pruned models is not destroyed during the clustering and fine-tuning process.
* Clustering schedule and callback, to provide more advanced training-time control of the clustering behaviour and progressive layer clustering throughout training.
* Improved training / fine-tuning by updating cluster-to-weight associations during the fine-tuning phase. This one is likely to be a good use case for having support for clustering schedule in the API.
* More cluster centroid initialization methods, e.g. k-means.

## Comparison with existing TFMOT tools

1. In contrast to the Keras quantization API &mdash; but similarly to pruning &mdash; clustering only acts on a single layer at a time.

2. Unlike pruning, weight clustering can only be applied to a pre-trained model, as the weights need to be known in order to determine the clusters.

## Platforms and Environments

Clustering is supported on all platforms that Keras is supported on.

## Best Practices

N/A

## Other

The following are essentially the same as for pruning, therefore we will not discuss them separately in this document:

* Performance Implications
* Dependencies
* Engineering Impact

Please refer to the [pruning design RFC](20200225-keras-pruning-design.md) for details about these topics.
