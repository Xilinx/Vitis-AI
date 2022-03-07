
.. meta::
   :keywords: New features
   :description: Release notes.
   :xlnxdocumentclass: Document
   :xlnxdocumenttype: Tutorials

.. _releasenotes-xfopencv:


Release notes
##############

The below section explains the new features added and also changes in the existing library along with the known issues.

-  `New features and functions <#pl-new>`_
-  `Known issues <#known-issues>`_

.. _pl-new:

New features and functions
============================

The below functions and pipelines are newly added into the library.

**Versal AI Engine additions** :

* blobFromImage
		Function used in many ML pre-processing tasks to do normalization and other tasks.
* Back to back filter2D with batch size three support  
		Application showcasing increasing throughput of single filter2D kernel, by doing 3, back-2-back filter2D achieving 555 FPS with PL datamovers.

**New Programmable Logic (PL) functions and features:**

* ISP pipeline and functions:
	* End to End Mono Image Processing (ISP) pipeline with CLAHE TMO
			Useful for ISP pipelines with monochrome sensors
	* RGB-IR along-with RGB-IR Image Processing (ISP) pipeline
			Useful for ISP pipelines with IR sensors
	* Global Tone Mapping (GTM) along with an ISP pipeline using GTM
			Adding to growing TMO (tone-mapping-operators) in the library for different quality and area tradeoff purposes: CLAHE, Local Tone Mapping, Quantization and Dithering


.. _known-issues:

Known issues
============

* Vitis GUI projects on RHEL83 and CEntOS82 may fail because of a lib conflict in the LD_LIBRARY_PATH setting. User needs to remove ${env_var:LD_LIBRARY_PATH} from the project environment settings for the function to build successfully.





















