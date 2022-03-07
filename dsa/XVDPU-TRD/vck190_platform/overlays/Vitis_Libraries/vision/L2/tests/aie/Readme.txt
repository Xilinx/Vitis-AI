Note: AIE PL testcases are not supported in GUI.

1. Open terminal
2. cd to any L2/tests/aie functions.
3. Do Environment settings:
	• bash
	• source < path-to-Vitis-installation-directory >/settings64.sh
		- Vitis™ 2021.1 or later version supported.
	• export DEVICE=< path-to-vck190 platform-directory >/< vck190 platform >.xpfm
	• source < path-to-XRT-installation-directory >/setup.sh
	• export SYSROOT=< path-to-versal-platform-sysroot >
	• export EDGE_COMMON_SW=< path-to-versal-rootfs-and-Image-files >
4. Open description.json and change value of "gui" to true.
5. Open Vitis GUI using command vitis and create application project for that aie function.
	• Do Environment settings for GUI:
	        - Add platform to repository: < path-to-vck190 platform-directory >/< vck190 platform >.xpfm
		- Sysroot path: < path-to-versal-platform-sysroot >
		- Root FS: < path-to-versal-platform-rootfs >
		- Kernel Image:  < path-to-versal-platform-kernel-Image >

6. Open host.cpp from GUI ProjName[xrt] -> src folder. If host.cpp has #include "graph.cpp" then replace graph.cpp with its absolute path i.e. the absolute path for graph.cpp present in GUI ProjName_aie[aiengine] -> src folder. To get absolute path, right click on graph.cpp and select Properties. Under Properties, location gives its absolute path.
7. Right click on ProjName[xrt] and select C/C++ Build Settings -> C/C++ Build -> Settings -> Tool Settings -> GCC Host Linker -> Libraries -> Library search path
8. Under Library search path add ${workspace_loc:${ProjName}/libs/xf_opencv/L1/lib/sw/aarch64-linux/}
9. We need aie_control_xrt.cpp to build AIE GUI project. This can be generated using Makefile flow.
10. Open a new terminal and cd to the selected AIE function, 
	• Repeat step 3.
	• make x86sim TARGET=hw_emu HOST_ARCH=aarch64
		- ignore make error or simulation failure
   This command generates aie_control_xrt.cpp at ./Work/ps/c_rts/
11. Copy aie_control_xrt.cpp from ./Work/ps/c_rts/and paste it in the GUI project folder where host.cpp is present.
12. Go to GUI project
	• Build Emulation-HW
	• Run hwemu-launch
