# iOS quickstart

To get started with TensorFlow Lite on iOS, we recommend exploring the following
example:

<a class="button button-primary" href="https://github.com/tensorflow/examples/tree/master/lite/examples/image_classification/ios">iOS
image classification example</a>

For an explanation of the source code, you should also read
[TensorFlow Lite iOS image classification](https://www.tensorflow.org/lite/models/image_classification/ios).

This example app uses
[image classification](https://www.tensorflow.org/lite/models/image_classification/overview)
to continuously classify whatever it sees from the device's rear-facing camera,
displaying the top most probable classifications. It allows the user to choose
between a floating point or
[quantized](https://www.tensorflow.org/lite/performance/post_training_quantization)
model and select the number of threads to perform inference on.

Note: Additional iOS applications demonstrating TensorFlow Lite in a variety of
use cases are available in [Examples](https://www.tensorflow.org/lite/examples).

## Add TensorFlow Lite to your Swift or Objective-C project

TensorFlow Lite offers native iOS libraries written in
[Swift](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/experimental/swift)
and
[Objective-C](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/experimental/objc).
To get started quickly writing your own iOS code, we recommend using our
[Swift image classification example](https://github.com/tensorflow/examples/tree/master/lite/examples/image_classification/ios)
as a starting point.

The sections below demonstrate how to add TensorFlow Lite Swift or Objective-C
to your project:

### CocoaPods developers

In your `Podfile`, add the TensorFlow Lite pod. Then, run `pod install`.

#### Swift

```ruby
use_frameworks!
pod 'TensorFlowLiteSwift'
```

#### Objective-C

```ruby
pod 'TensorFlowLiteObjC'
```

### Bazel developers

In your `BUILD` file, add the `TensorFlowLite` dependency to your target.

#### Swift

```python
swift_library(
  deps = [
      "//tensorflow/lite/experimental/swift:TensorFlowLite",
  ],
)
```

#### Objective-C

```python
objc_library(
  deps = [
      "//tensorflow/lite/experimental/objc:TensorFlowLite",
  ],
)
```

### Import the library

For Swift files, import the TensorFlow Lite module:

```swift
import TensorFlowLite
```

For Objective-C files, import the umbrella header:

```objectivec
#import "TFLTensorFlowLite.h"
```

Or, the module if you set `CLANG_ENABLE_MODULES = YES` in your Xcode project:

```objectivec
@import TFLTensorFlowLite;
```

Note: For CocoaPods developers who want to import the Objective-C TensorFlow
Lite module, you must also include `use_frameworks!` in your `Podfile`.
