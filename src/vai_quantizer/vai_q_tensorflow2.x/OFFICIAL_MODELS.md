# Production Example and Convergence Testing

One of the contribution requirements of a new technique is a training
example for using the technique on realistic tasks and models. At the same
time, the example will come with convergence tests to ensure that users can
continue to achieve the benefits described in the documentation.

The target task and model will be one of the ones available
in the [TensorFlow Official
Models](https://github.com/tensorflow/models/tree/master/official),
excluding trivial models and tasks such as MNIST.

This example and test will either live in TensorFlow Official Models
or in a fork of it, with the following process.

1. The Official Models team will review the proposal RFC and determine whether the
   example and tests for the technique will live in TF Official Models or a fork,
   based on the motivation for a user to use it and compatibility requirements.
     * TODO(TFMOT): provide examples on this criteria.

2. Reproducing the results from the RFC in the new environment will take time. While
   the contributor implements the library, in parallel they can start implementing
   the example and test by [building TFMOT pip packages from
   source](https://www.tensorflow.org/model_optimization/guide/install#installing_from_source).

3. While reproducing the results, review the the TensorFlow Official Models standards
   [here](https://github.com/tensorflow/models/wiki/Research-paper-code-contribution)
   and [here](https://github.com/tensorflow/models/wiki/Coding-guidelines). 
   The example and test that will be merged must abide by the standards.

4. Once the library for the technique is implemented, as well as the example and
   test, create a single pull request (PR) with the example and test.
     * First, TFMOT will review the PR, only approving after reproducing the results.
     * Next, if the example lives in TensorFlow Official Models, the Official Models
       team will review the PR and comment that it's okay for Official Models to 
       depend on a new TFMOT release with the technique.

5. After the approvals, TFMOT will create a stable release that includes the
   new technique.

6. If the example lives in Official Models, the minimum TFMOT version is bumped, and Official Models gives a final
   Github approval for merging the PR. Otherwise if the example lives in the fork, then the fork's minimum version will be
   bumped and TFMOT will give a final approval.

7. Final result: the technique is released in the TFMOT pip package with a production-level example and
   convergence test.

## Integration Details

TODO(tfmot): the fork of Official Models does not exist yet.

## Post-Integration Ownership
TODO(tfmot): link to ownership RFC.

Maintaining the example and ensuring the test continues to pass with tf-nightly
at TFMOT head is a part of owning the technique.

1. If a PR causes the tests to stop passing, TFMOT will revert the PR that is
   responsible. Any of the technique's owners should work with TFMOT to improve
   the process so that the owner can isolate and debug the issue themselves.

2. For any TFMOT release, all these tests must pass.

3. For PRs, first get an approval from TFMOT prior to requesting a review
   from the Official Models team.




