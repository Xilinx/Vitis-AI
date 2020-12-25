## Image Classification evaluation based on ILSVRC 2012 task

This binary evaluates the following parameters of TFLite models trained for the
[ILSVRC 2012 image classification task](http://www.image-net.org/challenges/LSVRC/2012/):

*   Native pre-processing latency
*   Inference latency
*   Top-K (1 to 10) accuracy values

The binary takes the path to validation images and labels as inputs, along with
the model and inference-specific parameters such as delegate and number of
threads. It outputs the metrics as a text proto to a file, similar to the
following:

```
num_runs: 300 # Total images evaluated
process_metrics {
  image_classification_metrics {
    pre_processing_latency {
      last_us: 8641
      max_us: 42357
      min_us: 2811
      sum_us: 2449340
      avg_us: 8164.4666666666662 # Avg Pre-processing latency in micro-seconds
    }
    inference_latency {
      last_us: 27979
      max_us: 40696
      min_us: 27811
      sum_us: 9142187
      avg_us: 30473.956666666665 # Avg Inference latency in micro-seconds
    }
    inference_metrics {
      num_inferences: 300
    }
    topk_accuracy_metrics {
      topk_accuracies: 0.7033333 # Top-1 accuracy
      topk_accuracies: 0.78
      topk_accuracies: 0.8333333
      topk_accuracies: 0.86
      topk_accuracies: 0.88
      topk_accuracies: 0.89
      topk_accuracies: 0.9033333
      topk_accuracies: 0.9033333
      topk_accuracies: 0.92
      topk_accuracies: 0.9266667 # Top-10 accuracy
    }
  }
}
```

To run the binary download the ILSVRC 2012 devkit
[see instructions](#downloading-ilsvrc) and run the
[`generate_validation_ground_truth` script](#ground-truth-label-generation) to
generate the ground truth labels.

## Parameters

The binary takes the following parameters:

*   `model_file` : `string` \
    Path to the TFlite model file.

*   `ground_truth_images_path`: `string` \
    The path to the directory containing ground truth images.

*   `ground_truth_labels`: `string` \
    Path to ground truth labels file. This file should contain the same number
    of labels as the number images in the ground truth directory. The labels are
    assumed to be in the same order as the sorted filename of images. See
    [ground truth label generation](#ground-truth-label-generation) section for
    more information about how to generate labels for images.

*   `model_output_labels`: `string` \
    Path to the file containing labels, that is used to interpret the output of
    the model. E.g. in case of mobilenets, this is the path to
    `mobilenet_labels.txt` where each label is in the same order as the output
    1001 dimension tensor.

*   `output_file_path`: `string` \
    The final metrics are dumped into `output_file_path` as a string-serialized
    instance of `tflite::evaluation::EvaluationStageMetrics`.

and the following optional parameters:

*   `blacklist_file_path`: `string` \
    Path to blacklist file. This file contains the indices of images that are
    blacklisted for evaluation. 1762 images are blacklisted in ILSVRC dataset.
    For details please refer to readme.txt of ILSVRC2014 devkit.

*   `num_images`: `int` (default=0) \
    The number of images to process, if 0, all images in the directory are
    processed otherwise only num_images will be processed.

*   `num_threads`: `int` (default=4) \
    The number of threads to use for evaluation. Note: This does not change the
    number of TFLite Interpreter threads, but shards the dataset to speed up
    evaluation.

The following optional parameters can be used to modify the inference runtime:

*   `num_interpreter_threads`: `int` (default=1) \
    This modifies the number of threads used by the TFLite Interpreter for
    inference.

*   `delegate`: `string` \
    If provided, tries to use the specified delegate for accuracy evaluation.
    Valid values: "nnapi", "gpu".

## Downloading ILSVRC

In order to use this tool to run evaluation on the full 50K ImageNet dataset,
download the data set from http://image-net.org/request.

## Ground truth label generation

The ILSVRC 2012 devkit `validation_ground_truth.txt` contains IDs that
correspond to synset of the image. The accuracy binary however expects the
ground truth labels to contain the actual name of category instead of synset
ids. A conversion script has been provided to convert the validation ground
truth to category labels. The `validation_ground_truth.txt` can be converted by
the following steps:

```
ILSVRC_2012_DEVKIT_DIR=[set to path to ILSVRC 2012 devkit]
VALIDATION_LABELS=[set to  path to output]

python tensorflow/lite/tools/accuracy/ilsvrc/generate_validation_labels.py \
--ilsvrc_devkit_dir=${ILSVRC_2012_DEVKIT_DIR} \
--validation_labels_output=${VALIDATION_LABELS}
```

## Running the binary

### On Android

(0) Refer to
https://github.com/tensorflow/tensorflow/tree/master/tensorflow/examples/android
for configuring NDK and SDK.

(1) Build using the following command:

```
bazel build -c opt \
  --config=android_arm \
  --cxxopt='--std=c++11' \
  --copt=-D__ANDROID_TYPES_FULL__ \
  --copt=-DSUPPORT_SELECTIVE_REGISTRATION \
  //tensorflow/lite/tools/evaluation/tasks/imagenet_image_classification:run_eval
```

(2) Connect your phone. Push the binary to your phone with adb push (make the
directory if required):

```
adb push bazel-bin/tensorflow/lite/tools/evaluation/tasks/imagenet_image_classification/run_eval /data/local/tmp
```

(3) Make the binary executable.

```
adb shell chmod +x /data/local/tmp/run_eval
```

(4) Push the TFLite model that you need to test. For example:

```
adb push mobilenet_quant_v1_224.tflite /data/local/tmp
```

(5) Push the imagenet images to device, make sure device has sufficient storage
available before pushing the dataset:

```
adb shell mkdir /data/local/tmp/ilsvrc_images && \
adb push ${IMAGENET_IMAGES_DIR} /data/local/tmp/ilsvrc_images
```

(6) Push the generated validation ground labels to device.

```
adb push ${VALIDATION_LABELS} /data/local/tmp/ilsvrc_validation_labels.txt
```

(7) Push the model labels text file to device.

```
adb push ${MODEL_LABELS_TXT} /data/local/tmp/model_output_labels.txt
```

(8) Run the binary.

```
adb shell /data/local/tmp/run_eval \
  --model_file=/data/local/tmp/mobilenet_quant_v1_224.tflite \
  --ground_truth_images_path=/data/local/tmp/ilsvrc_images \
  --ground_truth_labels=/data/local/tmp/ilsvrc_validation_labels.txt \
  --model_output_labels=/data/local/tmp/model_output_labels.txt \
  --output_file_path=/data/local/tmp/accuracy_output.txt \
  --num_images=0 # Run on all images.
```

### On Desktop

(1) Build and run using the following command:

```
bazel run -c opt \
  --cxxopt='--std=c++11' \
  -- \
  //tensorflow/lite/tools/evaluation/tasks/imagenet_image_classification:run_eval \
  --model_file=mobilenet_quant_v1_224.tflite \
  --ground_truth_images_path=${IMAGENET_IMAGES_DIR} \
  --ground_truth_labels=${VALIDATION_LABELS} \
  --model_output_labels=${MODEL_LABELS_TXT} \
  --output_file_path=/tmp/accuracy_output.txt \
  --num_images=0 # Run on all images.
```
