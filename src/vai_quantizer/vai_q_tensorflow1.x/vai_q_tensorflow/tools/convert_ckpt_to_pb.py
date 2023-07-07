import tensorflow as tf

ckpt_dir_meta="./ckptdir/ResNet-L50.meta"
ckpt_dir="./ckptdir"
output_name="./resnet_50_model.pb"

init=tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    saver=tf.train.import_meta_graph(ckpt_dir_meta)
    ckpt=tf.train.get_checkpoint_state(ckpt_dir)
    saver.restore(sess,ckpt.model_checkpoint_path)

    graph=tf.graph_util.convert_variables_to_constants(sess,sess.graph_def,['prob'])
    # tf.train.write_graph(graph,'./',output_name,as_text=False)
    tf.train.write_graph(graph,'./',output_name,as_text=True)
