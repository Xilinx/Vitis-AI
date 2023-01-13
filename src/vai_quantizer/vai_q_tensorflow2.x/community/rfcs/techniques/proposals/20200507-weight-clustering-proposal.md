# Keras Weight Clustering API

| Status        | Accepted                                       |
:---------------|:----------------------------------------------|
| **Author(s)** | Mohamed Nour Abouelseoud (mohamednour.abouelseoud@arm.com), Aron Virginas-Tar (aron.virginas-tar@arm.com), Anton Kachatkou (anton.kachatkou@arm.com)|
| **Sponsor**   | Alan Chiao (alanchiao@google.com)             |
| **Updated**   | 2020-05-07                                    |

We propose to implement the technique of weight clustering on top of the Keras API.

## Motivation and Overview

Clustering, or weight sharing, brings improvements via model compression. The technique revolves around reducing the number of unique weights in a model, thus facilitating storage and over-the-wire size reduction. Further framework and hardware support can also unlock memory footprint improvements.

Clustering can be combined with pruning and quantization, reducing the network’s storage requirements even further &mdash; see the paper linked [below](#proposed-design) for more details.

## Experimental Results

The table below summarizes the compression rate and top-1 accuracy achieved when clustering the weights of different models using linear centroid initialization.

<table>
  <tr>
    <th rowspan="2">Model</th>
    <th rowspan="2">Dataset</th>
    <th colspan="2">Original</th>
    <th colspan="4">Clustered</th>
  </tr>
  <tr>
  <th>Top-1 accuracy (%)</th>
    <th>Size of compressed .tflite (MB)</th>
    <th>Configuration</th>
    <th># of clusters</th>
    <th>Top-1 accuracy</th>
    <th>Size of compressed .tflite</th>
  </tr>
  <tr>
    <td>ConvNet</td>
    <td>MNIST</td>
    <td>99.40</td>
    <td>0.57</td>
    <td>Full model</td>
    <td>32</td>
    <td>98.78</td>
    <td>0.09</td>
  </tr>
  <tr>
    <td rowspan="4">MobileNetV1</td>
    <td rowspan="4">ImageNet</td>
    <td rowspan="4">70.60</td>
    <td rowspan="4">14.98</td>
  </tr>
  <tr>
    <td>Selective, last Conv2D layer</td>
    <td>32</td>
    <td>69.64</td>
    <td>11.90</td>
  </tr>
  <tr>
    <td>Selective, last 3 Conv2D layers</td>
    <td>256, 256, 32</td>
    <td>67.41</td>
    <td>8.77</td>
  </tr>
  <tr>
    <td>Full model (except DepthwiseConv2D layers)</td>
    <td>32</td>
    <td>64</td>
    <td>2.17</td>
  </tr>  
  <tr>
    <td rowspan="3">MobileNetV2</td>
    <td rowspan="3">ImageNet</td>
    <td rowspan="3">72.29</td>
    <td rowspan="3">12.90</td>
  </tr>
  <tr>
    <td>Selective, last 3 Conv2D layers</td>  
    <td>256, 256, 32</td>
    <td>72.31</td>
    <td>7.00</td>
 </tr>
 <tr>
   <td>Full model (except DepthwiseConv2D layers)</td>
   <td>32</td>
   <td>69.33</td>
   <td>2.60</td>
  </tr>
  <tr>
    <td>DS-CNN-L</td>
    <td>Speech Commands v0.02</td>
    <td>95.03</td>
    <td>1.50</td>
    <td>Full model</td>
    <td>32</td>
    <td>94.71</td>
    <td>0.30</td>
  </tr>
</table>

NOTE: *Size of compressed .tflite* refers to the size of the zipped .tflite file obtained from the model through the following process:
1. Serialize the Keras model into .h5 file
2. Convert the .h5 file into .tflite using `TFLiteConverter.from_keras_model_file()`
3. Compress the .tflite file into a zip

Zipping the .tflite file is relevant in this context in order to highlight the compression benefits. Clustering does not reduce the size of the weights per se, but it renders them suitable for compression via Huffman coding and similar techniques. In consequence, though the clustered .tflite file will have similar size to the non-clustered one, their zipped versions will greatly differ in size &mdash; as shown by the figures above.

### Known Limitations

Since weight clustering operates on a per-layer basis, applying batch normalisation folding or per-channel quantization on a clustered model in the TFLiteConverter may result in lower compression ratios than expected. This is due to these operations potentially increasing the number of unique weight values compared to the initial number of clusters.

## Proposed Approach

The weight clustering technique that forms the basis of our implementation has been described in the conference paper *Deep Compression: Compressing Deep Neural Networks with Pruning, Trained Quantization and Huffman Coding* by S. Han, H. Mao, and  W. J. Dally, which can be downloaded from [arXiv](https://arxiv.org/abs/1510.00149). Please refer to chapter 3, entitled *Trained Quantization and Weight Sharing*, for details about the clustering technique &mdash; since the paper is brief and freely available for download, we will not reproduce its content here.

## Contribution

The Machine Learning team from [Arm](https://www.arm.com/) (initially *Anton Kachatkou* and *Aron Virginas-Tar*) will be implementing this technique with the intention of being a production-level owner.
