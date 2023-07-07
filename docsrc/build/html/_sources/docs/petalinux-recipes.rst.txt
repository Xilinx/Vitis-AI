
Vitis AI Linux Recipes
======================

.. note:: The instructions refer to the sources found in the ``/src/petalinux_recipes`` folder within the Vitis™ AI repository.

There are two ways to install the Vitis AI libraries:

- To rebuild the system by configuring PetaLinux (Build-Time).
- To install VAI3.0 to the target leveraging a pre-built package (Run-Time). See the board setup instructions for details of the online installation :doc:`process <../docs/board_setup/vai_install_to_target>`.

Rebuilding the System by Configuring PetaLinux
----------------------------------------------

If users want to install VAI3.0 into the rootfs when generating system image with PetaLinux, users need to use the VAI3.0 recipes. You can obtain the recipes for VAI3.0 in the following two ways:

-  Using ``recipes-vitis-ai`` in this repo.
-  Upgrading PetaLinux eSDK.

Using recipes-vitis-ai
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. note::
     
   ``recipes-vitis-ai`` enables **Vitis flow by default**. If want to use the Vivado® flow, please comment the following line in ``recipes-vitis-ai/vart/vart_3.0.bb``

   ..  code-block:: bash

        #PACKAGECONFIG_append = " vitis"

1. Copy the ``recipes-vitis-ai`` folder to ``<petalinux project>/project-spec/meta-user/``

   ..  code-block:: bash

      cp Vitis-AI/src/petalinux_recipes/recipes-vitis-ai <petalinux project>/project-spec/meta-user/

2. Edit ``<petalinux project>/project-spec/meta-user/conf/user-rootfsconfig``
   file, appending the following lines:

   .. code-block::

         CONFIG_vitis-ai-library
         CONFIG_vitis-ai-library-dev
         CONFIG_vitis-ai-library-dbg

3. Source PetaLinux tool and run ``petalinux-config -c rootfs`` command. Select the following option.

   .. code-block::

         Select user packages --->
         Select [*] vitis-ai-library

   Then, save it and exit.

4. Run ``petalinux-build``.

   .. note:
      
      After you run the above successfully, the vitis-ai-library, VART3.0 and the dependent packages will all be installed into the rootfs image.

      If you want to compile the example on the target, please select the ``vitis-ai-library-dev`` and ``packagegroup-petalinux-self-hosted``. Then, recompile the system.

      If you want to use vaitracer tool, please select the ``vitis-ai-library-dbg``. And copy ``recipes-vai-kernel`` folder to ``<petalinux project>/project-spec/meta-user/``. Then, recompile the system.

   ..  code-block:: bash

       cp Vitis-AI/src/petalinux_recipes/recipes-vai-kernel <petalinux project>/project-spec/meta-user/

Using Upgrade PetaLinux eSDK
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Run the following commands to upgrade the PetaLinux.

.. code-block:: bash

     source <petalinux-v2022.2>/settings
     petalinux-upgrade -u ‘http://petalinux.xilinx.com/sswreleases/rel-v2022/sdkupdate/2022.2_update1/’ -p ‘aarch64’

Then, you can find ``vitis-ai-library_3.0.bb`` recipe in ``<petalinux project>/components/yocto/layers/meta-vitis-ai``.

For details about PetaLinux upgrading, refer to `PetaLinux Upgrade <https://docs.xilinx.com/r/en-US/ug1144-petalinux-tools-reference-guide/petalinux-upgrade-Option>`__.

.. note:: ``2022.2_update1`` will be released approximately 1 month after Vitis 3.0 release. The name of ``2022.2_update1`` may be changed. Modify it accordingly.
