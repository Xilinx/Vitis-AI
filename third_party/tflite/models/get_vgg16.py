import tensorflow as tf

# Import the model (see https://www.tensorflow.org/api_docs/python/tf/keras/applications/vgg16/VGG16)
tf_model = tf.keras.applications.vgg16.VGG16(
    include_top=True, weights='imagenet', input_tensor=None,
    input_shape=None, pooling=None, classes=1000,
    classifier_activation='softmax'
)

# Convert the model
converter = tf.lite.TFLiteConverter.from_keras_model(tf_model)
tflite_model = converter.convert()

# Save the model
with open('vgg16.tflite', 'wb') as f:
  f.write(tflite_model)
