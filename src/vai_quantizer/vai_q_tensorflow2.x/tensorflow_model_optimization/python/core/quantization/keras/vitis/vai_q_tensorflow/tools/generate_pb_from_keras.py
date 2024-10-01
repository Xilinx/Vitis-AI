import tensorflow as tf 
from tensorflow import keras as K
from tensorflow.python.tools import freeze_graph

import os

CKPT_FILE = 'float_model.ckpt'
INFER_FILE = 'inference_graph.pb'
FREEZE_FILE = 'freeze.pb'

os.environ["CUDA_VISIBLE_DEVICES"] = "1" 

config = tf.ConfigProto(
    gpu_options=tf.GPUOptions(
        visible_device_list="0", # specify GPU number
        allow_growth=True
    )
)

inputs = tf.compat.v1.placeholder(tf.float32, shape=[None, 512, 512 , 3], name='Input_image_2')
net = tf.compat.v1.layers.conv2d(inputs=inputs, filters=32, kernel_size=3, strides=1, use_bias=False, padding='same')
net = tf.compat.v1.layers.batch_normalization(inputs=net, training=False)
net = tf.nn.relu(net)
net = tf.compat.v1.layers.conv2d(inputs=net, filters=1, kernel_size=1, strides=1, use_bias=False, padding='same')
net = tf.compat.v1.layers.batch_normalization(inputs=net, training=False)                       
#net = tf.identity(net)
#net = K.activations.hard_sigmoid(net)
net = tf.nn.relu6(net + 3) * 0.16667

saver = tf.compat.v1.train.Saver()
with tf.compat.v1.Session(config=config) as sess:
    print(sess)
    sess.run(tf.compat.v1.initializers.global_variables())
    saver.save(sess, CKPT_FILE)
    print(' Saved checkpoint to %s' % CKPT_FILE, flush=True)

    tf.io.write_graph(sess.graph_def, './', INFER_FILE, as_text=False)
    print(' Saved binary inference graph to %s' % INFER_FILE)

    [print(n.name) for n in tf.get_default_graph().as_graph_def().node]
    #freeze_graph.freeze_graph(input_graph= INFER_FILE, input_saver='', input_binary=True, input_checkpoint=CKPT_FILE, output_node_names="Identity", restore_op_name='', filename_tensor_name='', output_graph=FREEZE_FILE, clear_devices=False, initializer_nodes='')
    #freeze_graph.freeze_graph(input_graph= INFER_FILE, input_saver='', input_binary=True, input_checkpoint=CKPT_FILE, output_node_names="Sigmoid", restore_op_name='', filename_tensor_name='', output_graph=FREEZE_FILE, clear_devices=False, initializer_nodes='')
    freeze_graph.freeze_graph(input_graph= INFER_FILE, input_saver='', input_binary=True, input_checkpoint=CKPT_FILE, output_node_names="mul", restore_op_name='', filename_tensor_name='', output_graph=FREEZE_FILE, clear_devices=False, initializer_nodes='')
    print(' Freeze graph to %s' % FREEZE_FILE)
