# Guidelines for Contributing a New Technique

The following are guidelines for contributing a complete technique to the
toolkit, for example, connection pruning.

We will be evolving these guidelines to make the process more effective, and
as we receive more contributions. For example, we are working on creating a
repository of training scripts for
different models and tasks aimed at simplifying technique validation
([issue](https://github.com/tensorflow/model-optimization/issues/133)),
making the project contributor-friendly
([issue](https://github.com/tensorflow/model-optimization/issues/131)), and
having reproducible results.


## Contribution Process

1. Before anything else, provide a proposal RFC that focuses on what would motivate a user
   to use this technique, rather than on the specifics of the API. If approved,
   you will be matched with a sponsor, following the
   [general TensorFlow RFC process](https://github.com/tensorflow/community/blob/master/governance/TF-RFCs.md).
   Consider the following:
   * The end to end user story. It's encouraged to consider how the technique fits with the other parts of the toolkit.
   * Experiment results for real world models that support the user story, with not only accuracy but also deployment metrics
     (e.g. model storage space, latency, memory, to mention a few).
   * Contributing requires ownership of the technique. Select the type of
     ownership. TODO(tfmot): link to ownership RFC.
   * TODO(tfmot): link to sample pruning proposal RFC.

2. In parallel:
   * Provide a design RFC under [model-optimization/community/rfcs](https://github.com/tensorflow/model-optimization/blob/master/community/rfcs). See the [TensorFlow process](https://github.com/tensorflow/community/blob/master/governance/TF-RFCs.md) for details.
     * When reasonable, the API should strive for similarity with existing techniques to provide the
       best user experience.
     * TODO(tfmot): link to sample pruning design RFC.
   * Start prototyping and building the library in a fork of TFMOT.

3. Upon approval of design RFC, upstream existing code into TFMOT
   and continue development there.

4. Implement the design RFC. In parallel, create a training script
   to reproduce the results in the RFC. This script
   will serve as an example to users and come with tests to ensure that
   results continue to be reproducible. See [this page](OFFICIAL_MODELS.md) for details.

5. Documentation and tutorials:
   * For a consistent user experience, these should strive for similarity with existing
     documentation where it makes sense.
     * TODO(tfmot): link to consistent user experience RFC.
   * Overview page that requires minimal end-user domain knowledge.
     * See [the one for quantization aware training](https://www.tensorflow.org/model_optimization/guide/quantization/training) as an example.
   * Colab end to end example that covers the single most critical path.
     * See [the one for quantization aware training](https://www.tensorflow.org/model_optimization/guide/quantization/training_example) as an example.
   * Comprehensive guide that covers all usage patterns and navigates users
     to the APIs for their use case.
     * See [the one for quantization aware training](https://www.tensorflow.org/model_optimization/guide/quantization/training_comprehensive_guide) as an example.

6. Packaging and release:
   * Releases are managed by the TensorFlow Model Optimization team. Work with
     them to produce and test pip package as well as generate API docs. To
     generate API docs locally, see instructions [here](https://github.com/tensorflow/model-optimization/blob/master/build_docs.py).
   * Days before the release date:
     * The API docs will not be checked in, but should be tested and
       ready to submit.
     * The other documentation will be be checked in, but not linked in the
       navigation bar. The colabs should use a stable TFMOT release, as opposed to
       a test release.

7. Collaborative blog post (optional)
   * samples: [pruning
     API](https://medium.com/tensorflow/tensorflow-model-optimization-toolkit-pruning-api-42cac9157a6a)
     and [post-training integer quantization](https://medium.com/tensorflow/tensorflow-model-optimization-toolkit-post-training-integer-quantization-b4964a1ea9ba)
