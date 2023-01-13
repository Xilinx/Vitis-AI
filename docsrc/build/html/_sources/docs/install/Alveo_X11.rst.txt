X11 Support for Running Vitis AI Docker with Alveo
====================================================

If you are running Vitis™ AI docker with Alveo™ card and want to use X11 support for graphics (for example, some demo applications in VART and Vitis AI Library for Alveo need to display images or video), add the following line into the *docker_run_params* variable definition in ``docker_run.sh`` script:

.. code-block::

    -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix -v $HOME/.Xauthority:/tmp/.Xauthority \

And after the docker starts up, run following command lines:

.. code-block::

    cp /tmp/.Xauthority ~/
    sudo chown vitis-ai-user:vitis-ai-group ~/.Xauthority

.. note:: Before running this script, please make sure you have a local X11 server running if you are using a Windows-based ssh terminal to connect to a remote server or you have run the **xhost +** command at a command terminal if you are using Linux with Desktop. Also, if you are using ssh to connect to the remote server, remember to enable *X11 Forwarding* option either with the Windows ssh tools setting or with *-X* options in ssh command line.

.. |trade|  unicode:: U+02223 .. TRADEMARK SIGN
   :ltrim:
.. |reg|    unicode:: U+000AE .. REGISTERED TRADEMARK SIGN
   :ltrim:
