import tensorflow as tf

from tensorflow.keras import layers

from tf_nndct.optimization.expanding import ExpandingRunner
import numpy as np

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
  input_spec = tf.TensorSpec((1, *input_shape), tf.float32)
  runner = ExpandingRunner(model, input_spec)
  expanded_model, _ = runner.expand(3)

  x = np.random.random(size=(1, *input_shape)).astype(np.float32)
  model.compile()
  expanded_model.compile()
  model_output = model(x)
  expanded_model_output = expanded_model(x)
  print("diff = ", model_output - expanded_model_output)
