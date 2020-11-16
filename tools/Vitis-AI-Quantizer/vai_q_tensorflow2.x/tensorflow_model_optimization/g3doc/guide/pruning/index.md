# Trim insignificant weights

This document provides an overview on model pruning to help you determine how it
fits with your use case.

*   To dive right into an end-to-end example, see the
    [Pruning with Keras](pruning_with_keras.ipynb) example.
*   To quickly find the APIs you need for your use case, see the
    [pruning comprehensive guide](comprehensive_guide.ipynb).

## Overview

Magnitude-based weight pruning gradually zeroes out model weights during the
training process to achieve model sparsity. Sparse models are easier to
compress, and we can skip the zeroes during inference for latency improvements.

This technique brings improvements via model compression. In the future,
framework support for this technique will provide latency improvements. We've
seen up to 6x improvements in model compression with minimal loss of accuracy.

The technique is being evaluated in various speech applications, such as
speech recognition and text-to-speech, and has been experimented on across
various vision and translation models.

### API Compatibility Matrix
Users can apply pruning with the following APIs:

*   Model building: `tf.keras` with only Sequential and Functional models
*   TensorFlow versions: TF 1.x for versions 1.14+ and 2.x.
    *   `tf.compat.v1` with a TF 2.X package and `tf.compat.v2` with a TF 1.X
        package are not supported.
*   TensorFlow execution mode: both graph and eager
*   Distributed training: `tf.distribute` with only graph execution

It is on our roadmap to add support in the following areas:

*   [Minimal Subclassed model support](https://github.com/tensorflow/model-optimization/issues/155)
*   [Framework support for latency improvements](https://github.com/tensorflow/model-optimization/issues/173)

## Results

### Image Classification

<figure>
  <table>
    <tr>
      <th>Model</th>
      <th>Non-sparse Top-1 Accuracy </th>
      <th>Sparse Accuracy </th>
      <th>Sparsity </th>
    </tr>
    <tr>
      <td rowspan=3>InceptionV3</td>
      <td rowspan=3>78.1%</td>
      <td>78.0%</td>
      <td>50%</td>
    </tr>
    <tr>
      <td>76.1%</td><td>75%</td>
    </tr>
    <tr>
      <td>74.6%</td><td>87.5%</td>
    </tr>
    <tr>
      <td>MobilenetV1 224</td><td>71.04%</td><td>70.84%</td><td>50%</td>
    </tr>
 </table>
</figure>

The models were tested on Imagenet.

### Translation

<figure>
  <table>
    <tr>
      <th>Model</th>
      <th>Non-sparse BLEU </th>
      <th>Sparse BLEU </th>
      <th>Sparsity </th>
    </tr>
    <tr>
      <td rowspan=3>GNMT EN-DE</td>
      <td rowspan=3>26.77</td>
      <td>26.86</td>
      <td>80% </td>
    </tr>
    <tr>
      <td>26.52</td><td>85%</td>
    </tr>
    <tr>
      <td>26.19</td><td>90%</td>
    </tr>
    <tr>
      <td rowspan=3>GNMT DE-EN</td>
      <td rowspan=3>29.47</td>
      <td>29.50</td>
      <td>80% </td>
    </tr>
    <tr>
      <td>29.24</td><td>85%</td>
    </tr>
    <tr>
      <td>28.81</td><td>90%</td>
    </tr>
 </table>
</figure>

The models use WMT16 German and English dataset with news-test2013 as the dev
set and news-test2015 as the test set.

## Examples

In addition to the [Prune with Keras](pruning_with_keras.ipynb)
tutorial, see the following examples:

* Train a CNN model on the MNIST handwritten digit classification task with
pruning:
[code](https://github.com/tensorflow/model-optimization/blob/master/tensorflow_model_optimization/python/examples/sparsity/keras/mnist/mnist_cnn.py)
* Train a LSTM on the IMDB sentiment classification task with pruning:
[code](https://github.com/tensorflow/model-optimization/blob/master/tensorflow_model_optimization/python/examples/sparsity/keras/imdb/imdb_lstm.py)

For background, see *To prune, or not to prune: exploring the efficacy of
pruning for model compression* [[paper](https://arxiv.org/pdf/1710.01878.pdf)].
