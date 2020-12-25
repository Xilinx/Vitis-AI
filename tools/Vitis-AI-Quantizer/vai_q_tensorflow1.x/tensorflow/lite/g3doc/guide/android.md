# Android quickstart

To get started with TensorFlow Lite on Android, we recommend exploring the
following example.

<a class="button button-primary" href="https://github.com/tensorflow/examples/tree/master/lite/examples/image_classification/android">Android
image classification example</a>

For an explanation of the source code, you should also read
[TensorFlow Lite Android image classification](https://www.tensorflow.org/lite/models/image_classification/android).

This example app uses
[image classification](https://www.tensorflow.org/lite/models/image_classification/overview)
to continuously classify whatever it sees from the device's rear-facing camera.
The application can run either on device or emulator.

Inference is performed using the TensorFlow Lite Java API. The demo app
classifies frames in real-time, displaying the top most probable
classifications. It allows the user to choose between a floating point or
[quantized](https://www.tensorflow.org/lite/performance/post_training_quantization)
model, select the thread count, and decide whether to run on CPU, GPU, or via
[NNAPI](https://developer.android.com/ndk/guides/neuralnetworks).

Note: Additional Android applications demonstrating TensorFlow Lite in a variety
of use cases are available in
[Examples](https://www.tensorflow.org/lite/examples).

## Build in Android Studio

To build the example in Android Studio, follow the instructions in
[README.md](https://github.com/tensorflow/examples/blob/master/lite/examples/image_classification/android/README.md).

## Create your own Android app

To get started quickly writing your own Android code, we recommend using our
[Android image classification example](https://github.com/tensorflow/examples/tree/master/lite/examples/image_classification/android)
as a starting point.

The following sections contain some useful information for working with
TensorFlow Lite on Android.

### Use the TensorFlow Lite AAR from JCenter

To use TensorFlow Lite in your Android app, we recommend using the
[TensorFlow Lite AAR hosted at JCenter](https://bintray.com/google/tensorflow/tensorflow-lite).

You can specify this in your `build.gradle` dependencies as follows:

```build
dependencies {
    implementation 'org.tensorflow:tensorflow-lite:0.0.0-nightly'
}
```

This AAR includes binaries for all of the
[Android ABIs](https://developer.android.com/ndk/guides/abis). You can reduce
the size of your application's binary by only including the ABIs you need to
support.

We recommend most developers omit the `x86`, `x86_64`, and `arm32` ABIs. This
can be achieved with the following Gradle configuration, which specifically
includes only `armeabi-v7a` and `arm64-v8a`, which should cover most modern
Android devices.

```build
android {
    defaultConfig {
        ndk {
            abiFilters 'armeabi-v7a', 'arm64-v8a'
        }
    }
}
```

To learn more about `abiFilters`, see
[`NdkOptions`](https://google.github.io/android-gradle-dsl/current/com.android.build.gradle.internal.dsl.NdkOptions.html)
in the Android Gradle documentation.

### Build TensorFlow Lite locally

In some cases, you might wish to use a local build of TensorFlow Lite. For
example, you may be building a custom binary that includes
[operations selected from TensorFlow](https://www.tensorflow.org/lite/guide/ops_select),
or you may wish to make local changes to TensorFlow Lite.

#### Install Bazel and Android Prerequisites

Bazel is the primary build system for TensorFlow. To build with Bazel, it and
the Android NDK and SDK must be installed on your system.

1.  Install the latest version of Bazel as per the instructions
    [on the Bazel website](https://bazel.build/versions/master/docs/install.html).
2.  The Android NDK is required to build the native (C/C++) TensorFlow Lite
    code. The current recommended version is 17c, which may be found
    [here](https://developer.android.com/ndk/downloads/older_releases.html#ndk-17c-downloads).
3.  The Android SDK and build tools may be obtained
    [here](https://developer.android.com/tools/revisions/build-tools.html), or
    alternatively as part of
    [Android Studio](https://developer.android.com/studio/index.html). Build
    tools API >= 23 is the recommended version for building TensorFlow Lite.

#### Configure WORKSPACE and .bazelrc

Run the `./configure` script in the root TensorFlow checkout directory, and
answer "Yes" when the script asks to interactively configure the `./WORKSPACE`
for Android builds. The script will attempt to configure settings using the
following environment variables:

*   `ANDROID_SDK_HOME`
*   `ANDROID_SDK_API_LEVEL`
*   `ANDROID_NDK_HOME`
*   `ANDROID_NDK_API_LEVEL`

If these variables aren't set, they must be provided interactively in the script
prompt. Successful configuration should yield entries similar to the following
in the `.tf_configure.bazelrc` file in the root folder:

```shell
build --action_env ANDROID_NDK_HOME="/usr/local/android/android-ndk-r17c"
build --action_env ANDROID_NDK_API_LEVEL="18"
build --action_env ANDROID_BUILD_TOOLS_VERSION="28.0.3"
build --action_env ANDROID_SDK_API_LEVEL="23"
build --action_env ANDROID_SDK_HOME="/usr/local/android/android-sdk-linux"
```

#### Build and Install

Once Bazel is properly configured, you can build the TensorFlow Lite AAR from
the root checkout directory as follows:

```sh
bazel build --cxxopt='-std=c++11' -c opt         \
  --fat_apk_cpu=x86,x86_64,arm64-v8a,armeabi-v7a \
  //tensorflow/lite/java:tensorflow-lite
```

This will generate an AAR file in `bazel-genfiles/tensorflow/lite/java/`. Note
that this builds a "fat" AAR with several different architectures; if you don't
need all of them, use the subset appropriate for your deployment environment.
From there, there are several approaches you can take to use the .aar in your
Android Studio project.

##### Add AAR directly to project

Move the `tensorflow-lite.aar` file into a directory called `libs` in your
project. Modify your app's `build.gradle` file to reference the new directory
and replace the existing TensorFlow Lite dependency with the new local library,
e.g.:

```
allprojects {
    repositories {
        jcenter()
        flatDir {
            dirs 'libs'
        }
    }
}

dependencies {
    compile(name:'tensorflow-lite', ext:'aar')
}
```

##### Install AAR to local Maven repository

Execute the following command from your root checkout directory:

```sh
mvn install:install-file \
  -Dfile=bazel-genfiles/tensorflow/lite/java/tensorflow-lite.aar \
  -DgroupId=org.tensorflow \
  -DartifactId=tensorflow-lite -Dversion=0.1.100 -Dpackaging=aar
```

In your app's `build.gradle`, ensure you have the `mavenLocal()` dependency and
replace the standard TensorFlow Lite dependency with the one that has support
for select TensorFlow ops:

```
allprojects {
    repositories {
        jcenter()
        mavenLocal()
    }
}

dependencies {
    implementation 'org.tensorflow:tensorflow-lite-with-select-tf-ops:0.1.100'
}
```

Note that the `0.1.100` version here is purely for the sake of
testing/development. With the local AAR installed, you can use the standard
[TensorFlow Lite Java inference APIs](inference.md) in your app code.
