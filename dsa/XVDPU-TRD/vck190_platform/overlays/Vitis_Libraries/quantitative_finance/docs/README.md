Document Readme
=========================

## Generation Guide

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

## License

Licensed using the `Apache 2.0 license <https://www.apache.org/licenses/LICENSE-2.0>`_.

    Copyright 2019 Xilinx, Inc.
    
    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at
    
        http://www.apache.org/licenses/LICENSE-2.0
    
    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.


## Trademark Notice

Xilinx, the Xilinx logo, Artix, ISE, Kintex, Spartan, Virtex, Zynq, and other designated brands included herein are trademarks of Xilinx in the United States and other countries. All other trademarks are the property of their respective owners.
