#
# © Copyright 2023 Advanced Micro Devices, Inc. All rights reserved.
#
# This file contains confidential and proprietary information of Advanced
# Micro Devices, Inc. and is protected under U.S. and international
# copyright and other intellectual property laws.
#
# DISCLAIMER This disclaimer is not a license and does not grant any rights
# to the materials distributed herewith. Except as otherwise provided in a
# valid license issued to you by Xilinx, and to the maximum extent
# permitted by applicable law: (1) THESE MATERIALS ARE MADE AVAILABLE “AS
# IS” AND WITH ALL FAULTS, AND XILINX HEREBY DISCLAIMS ALL WARRANTIES AND
# CONDITIONS, EXPRESS, IMPLIED, OR STATUTORY, INCLUDING BUT NOT LIMITED TO
# WARRANTIES OF MERCHANTABILITY, NON-INFRINGEMENT, OR FITNESS FOR ANY
# PARTICULAR PURPOSE; and (2) Xilinx shall not be liable (whether in
# contract or tort, including negligence, or under any other theory of
# liability) for any loss or damage of any kind or nature related to,
# arising under or in connection with these materials, including for any
# direct, or any indirect, special, incidental, or consequential loss or
# damage (including loss of data, profits, goodwill, or any type of loss or
# damage suffered as a result of any action brought by a third party) even
# if such damage or loss was reasonably foreseeable or Xilinx had been
# advised of the possibility of the same.
#
# CRITICAL APPLICATIONS Xilinx products are not designed or intended to be
# fail-safe, or for use in any application requiring fail-safe performance,
# such as life-support or safety devices or systems, Class III medical
# devices, nuclear facilities, applications related to the deployment of
# airbags, or any other applications that could lead to death, personal
# injury, or severe property or environmental damage (individually and
# collectively, “Critical Applications”).  Customer assumes the sole risk
# and liability of any use of Xilinx products in Critical Applications,
# subject only to applicable laws and regulations governing limitations on
# product liability.
#
# THIS COPYRIGHT NOTICE AND DISCLAIMER MUST BE RETAINED AS PART OF THIS
# FILE AT ALL TIMES.
#
import numpy as np
import tensorflow as tf

from PIL import Image
from OnBoard import SummaryCollector


def demo_add_graph_wego_tf2(logdir):
    """OnBoard API Demo for WeGO-TF2.x, introduction of how to use OnBoard
    when you run a model. It would generate an event file in `logdir` by your
    defination, you can use exactly the same command of TensorBoard to obtain
    the visualize results: `tensorboard --logdir=same/as/logdir`.
    """
    from tensorflow.compiler.vitis_vai.python.vai_convert import create_wego_model
    from tensorflow.compiler import vitis_vai

    # Step 1. Create a collector of OnBoard and define `logdir`.
    collector = SummaryCollector(log_dir=logdir)
    
    # Step 2. Load model and compile it by using WeGO.
    model_path = "./models/resnet50_tf2/quantized.h5"
    wego_model = create_wego_model(model_path)
    
    # Step 3. Call `add_graph` to visualize the model graph, you should provide
    # `model`. You needn't provide `input_to_model` param, it is no need in TF2
    # framework and it would print the model graph on screen if set `verbose`.
    collector.add_graph(wego_model, input_to_model=None, verbose=False)

    # Final. Invoke `collector.close()` in the end.
    collector.close()

def demo_add_graph_tf2(logdir):
    """OnBoard API Demo for TorchVision models, samely as the usage for WeGO
     model. OnBoard is not depend on WeGO, you can use it when you run the 
     original Tensorflow 2.x models. The usage of them are exactly the same.
    """
    # Step 1. Create a collector of OnBoard and define `logdir`.
    collector = SummaryCollector(log_dir=logdir)

    # Step 2. Load model from Tensorflow hub or by your own path.
    # model_url = "https://storage.googleapis.com/tensorflow/keras-applications/mobilenet_v2/mobilenet_v2_weights_tf_dim_ordering_tf_kernels_1.0_224_no_top.h5"
    # model_path = tf.keras.utils.get_file("mobilenet_v2_weights_tf_dim_ordering_tf_kernels_1.0_224_no_top.h5", model_url)
    # model = tf.keras.applications.MobileNetV2(weights=model_path, include_top=False)
    model_path = "./models/resnet50_tf2/resnet_50.h5"
    model = tf.keras.models.load_model(model_path)
    
    # Step 3. Call `add_graph` to visualize the model graph, you should provide
    # `model`. You needn't provide `input_to_model` param, it is no need in TF2
    # framework and it would print the model graph on screen if set `verbose`.
    collector.add_graph(model, input_to_model=None, verbose=False)

    # Final. Invoke `collector.close()` in the end.
    collector.close()


def demo_add_text(logdir):
    """
    OnBoard API Demo for adding images to visualize in TensorBoard.
    """
    # You can use the add_text() function to add text and create an event file supported by TensorBoard.
    # Step 1. Create a collector of OnBoard and define `logdir`.
    collector = SummaryCollector(log_dir=logdir)
    # Step 2. Prepare the text you want to visualize.
    my_text="Hello world!"
    # Step 3. Call `add_text` to add text and create an event file supported by TensorBoard.
    collector.add_text('Hello',my_text)

    # Markdown format is also supported.
    md_text = \
    '''
### Markdown format is also supported!

You can use Markdown syntax to enhance your text and display code.

For example, you can use the following syntax to display a code block:

```python
def hello_world():
    print("Hello, World!")

hello_world()
```

Or, you can create a table:

| Column 1 | Column 2 |
| --- | --- |
| Data 1 | Data 2 |
| Data 3 | Data 4 |

    '''
    collector.add_text('MD text', md_text)

    # Step 4. Add platform information.
    collector.add_platform_info()
    # Step 5. Execute user-defined commands and collect the result.
    collector.user_command_result('whoami')

    # Final. Invoke `collector.close()` in the end.
    collector.close()

def demo_add_image(logdir):
    """
    OnBoard API Demo for adding images to visualize in TensorBoard.
    """
    # Step 1. Create a SummaryCollector object
    collector = SummaryCollector(log_dir=logdir)

    # Classification example
    # OnBoard supports TensorFlow tensors, PyTorch tensors, and NumPy arrays as image inputs.
    # For multiple inputs, we recommend using lists for tensors, tags, and bounding boxes (if needed)

    # Additionally, you can include a single image using the following code : 
    # collector.add_image("Samoyed:0.801775", "./img/dog.jpg", dataformats="HWC").

    # Step 2. Prepare input tensors and tags as lists.
    image = Image.open("./img/dog.jpg")
    img_tensor = np.array(image)
    tf_tensor = tf.convert_to_tensor(img_tensor)
    tensors = [tf_tensor, tf_tensor]

    tags = ["Samoyed:0.801775","Samoyed:0.801775"]
    collector.add_image(tags, tensors, dataformats="HWC")
    
    # Detection example with bounding boxes
    
    tags = ["detection_example", "detection_example"]
    box1 = np.array([[166.221, 35.7361, 1452.19, 1132.58]])
    box2 = np.array([[166.264, 35.7564, 1452.43, 1132.97], [239.652, 36.2654, 602.778, 395.646]])
    boxes = [box1, box2]
    collector.add_image(tags, tensors, boxes, dataformats="HWC")
    
    # Final. Invoke `collector.close()` in the end.
    collector.close()


if __name__=='__main__':
    demo_add_graph_wego_tf2("demo_tf2")
    demo_add_graph_tf2("demo_tf2")
    demo_add_image("demo_tf2")
    demo_add_text("demo_tf2")
