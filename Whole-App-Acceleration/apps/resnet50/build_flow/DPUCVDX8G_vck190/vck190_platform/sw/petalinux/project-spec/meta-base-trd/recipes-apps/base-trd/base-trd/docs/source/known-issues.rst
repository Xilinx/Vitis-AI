Known Issues
============
#. The APM, CPU utilization and power graph plots result in high CPU
   utilization. If the plots are disabled, the CPU utilization is reduced. This
   can be verified by running the top command from a terminal.

#. The monitor should show a blue standby screen after boot. If that is not the
   case, re-plug the HDMI cable and the blue standby screen should appear.

#. The gstreamer pipeline involving USB may sometimes fail with the following
   error. In this case click the rectangular icon to interrupt the kernel then
   select ‘Kernel’ --> ‘Restart Kernel and Run All Cells’ from the top menu bar
   to restart the pipeline.

   .. code-block:: bash

        ERROR: from element /GstPipeline:pipeline0/GstMediaSrcBin:mediasrcbin0/GstV4l2Src:v4l2src0: Cannot identify device '/dev/video3'.
        Additional debug info:
	../../../git/sys/v4l2/v4l2_calls.c(610): gst_v4l2_open (): /GstPipeline:nb3/GstMediaSrcBin:mediasrcbin0/GstV4l2Src:v4l2src0:
	system error: No such file or directory

#. If vivid or USB webcam is selected as video source, the following gstreamer
   error message is printed on the serial console. It is benign and can be
   ignored.

   .. code-block:: bash

        ** (python3:1404): CRITICAL **: 11:28:06.903: gst_video_info_from_caps: assertion 'gst_caps_is_fixed (caps)' failed
        0:00:02.146225828  1404 0xaaab03b29060 ERROR            mediasrcbin gstmediasrcbin.c:285:get_media_bus_format: Gst Fourcc 64205312 not handled

#. Enabling the primary plane on the Video Mixer by default results in a
   bandwidth utilization of 2GB. A patch is applied to disable the mixer primary
   plane by default. To enable/disable the primary plane through module_param
   and devmem use the following commands. To render a test pattern on the
   display using a utility like modetest, the primary plane should be enabled.

   * Enable:

   .. code-block:: bash

	echo Y > /sys/module/xlnx_mixer/parameters/mixer_primary_enable

   * Disable:

   .. code-block:: bash

	devmem 0xa0070040 32 0x0
	echo N > /sys/module/xlnx_mixer/parameters/mixer_primary_enable
