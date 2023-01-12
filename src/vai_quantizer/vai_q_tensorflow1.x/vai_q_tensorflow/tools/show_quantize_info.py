import tensorflow as tf

tf.app.flags.DEFINE_string('pb_file', '', 'input pb file')
FLAGS = tf.app.flags.FLAGS

if not FLAGS.pb_file:
  print("Usage: python show_quantize_info.py --pb_file deploy_model.pb")
  exit()

graphdef = tf.GraphDef()
graphdef.ParseFromString(tf.gfile.FastGFile(FLAGS.pb_file, "rb").read())
for node in graphdef.node:
  print("Op: {}, Type: {}".format(node.name, node.op))
  for key in node.attr:
    if key in ['ipos', 'opos', 'wpos', 'bpos']:
      print("  {}: bit_width: {} quantize_pos: {}".format(key, node.attr[key].list.i[0], node.attr[key].list.i[1]))
