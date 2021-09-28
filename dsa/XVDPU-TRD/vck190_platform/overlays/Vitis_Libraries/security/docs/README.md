Document Readme
=========================

## Gneration Guide

### Environment Variables

The following environment variables must be set before using the Makefile
to compile the document:

+ `HTML_DEST_DIR` points to the directory in which the HTML will be
  installed to.
+ `PATH` should include directory of `doxyrest` binary.

### Makefile targets

The top Makefile contains the following targets, each depending on the previous
one.

+ **xml**: call `doxygen` to extract the inline doc from C++ headers.
+ **rst**: call `doxyrest` to yield reStructuredText files from XML files.
+ **html**: call `sphinx` to generate HTML from generated reStructuredText
  files and manually written ones.
+ **install**: copy the HTML files to `$HTML_DEST_DIR`.

## Steps to update the doc

* Clone the `gh-pages` branch to a directory
* Set `HTML_DEST_DIR` to that directory
* Run `make install` in this folder.
* Go to `$HTML_DEST_DIR`, check in, and push to github.

## Copyright and Trademark Notice; Disclaimer

(c) Copyright 2019 Xilinx, Inc.

Xilinx, the Xilinx logo, Artix, ISE, Kintex, Spartan, Virtex, Zynq, and other designated brands included herein
are trademarks of Xilinx in the United States and other countries.
All other trademarks are the property of their respective owners.

DISCLAIMER
The information disclosed to you hereunder (the “Materials”) is provided solely for the selection and use of
Xilinx products. To the maximum extent permitted by applicable law: (1) Materials are made available "AS IS"
and with all faults, Xilinx hereby DISCLAIMS ALL WARRANTIES AND CONDITIONS, EXPRESS, IMPLIED, OR
STATUTORY, INCLUDING BUT NOT LIMITED TO WARRANTIES OF MERCHANTABILITY, NON-INFRINGEMENT,
OR FITNESS FOR ANY PARTICULAR PURPOSE; and (2) Xilinx shall not be liable (whether in contract or tort,
including negligence, or under any other theory of liability) for any loss or damage of any kind or nature
related to, arising under, or in connection with, the Materials (including your use of the Materials), including
for any direct, indirect, special, incidental, or consequential loss or damage (including loss of data, profits,
goodwill, or any type of loss or damage suffered as a result of any action brought by a third party) even if
such damage or loss was reasonably foreseeable or Xilinx had been advised of the possibility of the same.
Xilinx assumes no obligation to correct any errors contained in the Materials or to notify you of updates to
the Materials or to product specifications. You may not reproduce, modify, distribute, or publicly display the
Materials without prior written consent. Certain products are subject to the terms and conditions of Xilinx’s
limited warranty, please refer to Xilinx’s Terms of Sale which can be viewed at
http://www.xilinx.com/legal.htm#tos; IP cores may be subject to warranty and support terms contained in a
license issued to you by Xilinx. Xilinx products are not designed or intended to be fail-safe or for use in any
application requiring fail-safe performance; you assume sole risk and liability for use of Xilinx products in
such critical applications, please refer to Xilinx’s Terms of Sale which can be viewed at
http://www.xilinx.com/legal.htm#tos.

