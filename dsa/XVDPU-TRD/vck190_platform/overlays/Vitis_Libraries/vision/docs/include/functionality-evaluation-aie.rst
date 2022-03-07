Evaluating the Functionality
=============================

You can build the kernels and test the functionality through x86 simulation, cycle
accurate aie simulation, hw emulation or hw run on the board. Use the following
commands to setup the basic environment:

.. _x86_simulation:

x86 Simulation
--------------

Please refer to `x86 Functional Simulation`_ section in Vitis Unified Software Development Platform 2021.2 Documentation. For host code development, please refer to `Programming the PS Host Application`_ section

.. _x86 Functional Simulation: https://www.xilinx.com/html_docs/xilinx2021_1/vitis_doc/simulate_graph_application.html#uqf1619792614896
.. _Programming the PS Host Application: https://www.xilinx.com/html_docs/xilinx2021_1/vitis_doc/program_ps_host_application.html#ykt1590616160037

.. _aie_simulation:

AIE Simulation
--------------

Please refer to `AIE Simulation`_ section in Vitis Unified Software Development Platform 2021.2 Documentation. For host code development, please refer to `Programming the PS Host Application`_ section.

.. _AIE Simulation: https://www.xilinx.com/html_docs/xilinx2021_1/vitis_doc/simulate_graph_application.html#yql1512608436352

.. _hw_emulation:

HW emulation
------------

Please refer to `Programming the PS Host Application`_ section in Vitis Unified Software Development Platform 2021.2 Documentation.


Testing on HW
-------------

After the build for hardware target completes, sd_card.img file will be generated in the build directory. 

1. Use a software like Etcher to flash the sd_card.img file on to a SD Card. 
2. After flashing is complete, insert the SD card in the SD card slot on board and power on the board.
3. Use Teraterm to connect to COM port and wait for the system to boot up.
4. After the boot up is done, goto /media/sd-mmcblk0p1 directory and run the executable file.

Please refer to `hw_run`_ section in Vitis Unified Software Development Platform 2021.2 Documentation.

.. _hw_run: https://www.xilinx.com/html_docs/xilinx2021_1/vitis_doc/running_hw_app.html#lwu1600468728254