Branching / Tagging Strategy
============================

Each updated release of Vitis™ AI is pushed directly to `master <https://github.com/Xilinx/Vitis-AI/tree/master>`__ on the release day. In addition, at that time, a tag is created for the repository; for example, see the tag for `v3.0 <https://github.com/Xilinx/Vitis-AI/tree/v3.0>`__.

Following the release, the tagged version remains static, and additional inter-version updates are pushed to the master branch. Thus, the master branch is always the latest release and will have the latest fixes and documentation. The branch associated with a specific release (which will be “master” during the lifecycle of that release) will become a branch at the time of the next release.

Similarly, if you use a previous version of Vitis AI, the branch associated with that previous revision will contain updates to the tagged release for that same version. For instance, in the case of release 2.0, the `branch <https://github.com/Xilinx/Vitis-AI/tree/2.0>`__ contains updates that the `tag <https://github.com/Xilinx/Vitis-AI/tree/v2.0>`__ does not.

The diagram below depicts the overall workflow:

.. image:: ../reference/images/branching_strategy.PNG
