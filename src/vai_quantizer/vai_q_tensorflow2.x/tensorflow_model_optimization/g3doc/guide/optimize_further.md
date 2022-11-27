# Optimize further

When pre-optimized models and post-training tools do not satisfy your use case,
the next step is to try the different training-time tools.

Training time tools piggyback on the model's loss function over the training
data such that the model can "adapt" to the changes brought by the optimization
technique.

The starting point to use our training APIs is a Keras training script, which
can be optionally initialized from a pre-trained Keras model to further fine
tune.


Training time tools available for you to try:

* [Weight pruning](./pruning/)
* [Quantization] (./quantization/training)
