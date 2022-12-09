<table class="sphinxhide">
 <tr>
   <td align="center"><img src="https://raw.githubusercontent.com/Xilinx/Image-Collateral/main/xilinx-logo.png" width="30%"/><h1>Vitis AI</h1><h0>Adaptable & Real-Time AI Inference Acceleration</h0>
   </td>
 </tr>
</table>


<br />

As of the Vitis AI 3.0 release, we have migrated the historic Github repository documentation to [Github.IO](https://xilinx.github.io/Vitis-AI/).

The advantages are:

- Improved quality and usefulness of documentation
- Enhanced look and feel intended to improve the user experience
- ers working on air-gapped machines will be able to use the HTML versions of the documentation directly while working offline

To accomplish this, source documentation files in .rst format have been consolidated into the 3.0 release branch /docsrc directory which is then directly used as the source repository to compile the HTML version of the documentation.  The resulting HTML can be found in the /docs directory and can be used offline immediately once the repository has been cloned by the user.  The HTML documentation in the /docs folder is identical to what will be published on Github.IO.

Should you have a need to rebuild the offline HTML documentation, simply create a sphinx-build environment in Anaconda, navigate to the /docsrc directory and execute the command:

```text
make github
```

Once the build process has completed, you can find the compiled HTML documentation in the /docs folder.

<br /> <br />
