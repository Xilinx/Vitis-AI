# Sphinx Template for GitHub based HTML documentation

**10/09/2020** - Minor bug fixes and enhancements.
*	Updated layout.html to fix whitespace issue with copyright footer.
*	Updated layout.html to fix spacing in bulleted lists.

**03/06/2020** - Imported new stylesheets developed by the Web team
*	New css and js files added to support Xilinx look & feel
*	Updated layout.html to accomodate the header, footer and other elements on the page.
*	Moved the footer information from footer.html file to layout.html file to accomodate the Xilinx footer.
*	Updated conf.py to generate last modified date on all HTML pages.

**02/24/2020** - Udpated look & feel elements
* Updates themes.css file to remove All Caps fonts style for headings.
* Updated conf.py file to remove 'View on GitHub' link on the left navigation .
* Updated layout.html to remove the meta tags. Meta tags are now expected to be added to the index.rst file of each library. For example:

```
.. meta::
   :keywords: Vitis, Library, <library name>, Xilinx
   :description: <Provide a brief description of the file>
   :xlnxdocumentclass: Document
   :xlnxdocumenttypes: Tutorials

```

**10/04/2019** - Updated as per suggestions from Daniel
* Left align Xilinx logo.
* Shift over the “Developers , Support, Forums” to align with the end of the left column.
* Set a max char width for the left column so it isn’t so wide.
* Reduce font size of the copyright footer text. 

**09/24/2019** - Initial version
* Xilinx.com branding and look & feel shared by the Web team.
* references to .js and .css files used by Xilinx.com.
* meta data that can help users search these pages using the Xilinx.com search. 
* Link to GitHub source file.
