import tensorflow as tf

from tensorflow.keras import layers
from tf_nndct.optimization.expanding import expand_and_export


keras = tf.keras

def mnist_convnet():
  num_classes = 10
  input_shape = (28, 28, 1)

  model = keras.Sequential([
    keras.Input(shape=input_shape),
    layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
    layers.BatchNormalization(),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Flatten(),
    layers.Dropout(0.5),
    layers.Dense(num_classes, activation="softmax"),
  ])
  return model, input_shape


if __name__ == "__main__":
  model, input_shape = mnist_convnet()
  expand_and_export("mnist", model, tf.TensorSpec((1, *input_shape), tf.float32), "test_results", [4, 8], ["Input"], ["Identity"])
