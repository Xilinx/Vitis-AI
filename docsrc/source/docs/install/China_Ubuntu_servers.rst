:orphan:

Access to Ubuntu Mirrors from within China
==========================================

Vitis |trade| AI Docker images leverage Ubuntu 20.04. In your Ubuntu installation, the file **/etc/apt/sources.list** specifies the default server location for Ubuntu packages. For example:

.. code-block::

     deb http://us.archive.ubuntu.com/ubuntu/ focal universe

You can see that the hostname “archive.ubuntu.com” resolves to servers located within the United States. When building the Vitis AI Docker image, whether for CPU-only or GPU accelerated containers Docker will attempt to pull from US servers. As a result, users accessing from China will generally experience slow download speeds.

Prior to building the Vitis AI Docker image it is recommended that you modify **/etc/apt/sources.list** and the vitis-ai-gpu.Dockerfile.

In the Vitis AI .DockerFile, change the first instances of apt-get update and apt-get install.

From:

.. code-block::

     RUN chmod 1777 /tmp \
       .......
       .......
       && apt-get update -y \
       && apt-get install -y --no-install-recommends \

To:

.. code-block::

     RUN sed --in-place --regexp-extended "s/(\/\/)(archive\.ubuntu)/\1cn.\2/" /etc/apt/sources.list && apt-get update && apt-get install -y --no-install-recommends \

Next, modify your Ubuntu **/etc/apt/sources.list** to point to mirrors located in China. The following Tsinghua University (Beijing) mirrors are recommended:

.. caution:: Backup **/etc/apt/sources.list** prior to making this change.

.. code-block::

     deb https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ focal main restricted universe multiverse
     deb-src https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ focal main restricted universe multiverse
     deb https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ focal-updates main restricted universe multiverse
     deb-src https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ focal-updates main restricted universe multiverse
     deb https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ focal-backports main restricted universe multiverse
     deb-src https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ focal-backports main restricted universe multiverse
     deb https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ focal-security main restricted universe multiverse
     deb-src https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ focal-security main restricted universe multiverse
     deb https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ focal-proposed main restricted universe multiverse
     deb-src https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ focal-proposed main restricted universe multiverse

In addition, multiple alternative mirrors are `here <https://momane.com/change-ubuntu-20-04-source-to-china-mirror>`__.

.. |trade|  unicode:: U+02122 .. TRADEMARK SIGN
   :ltrim:
.. |reg|    unicode:: U+000AE .. REGISTERED TRADEMARK SIGN
   :ltrim:
