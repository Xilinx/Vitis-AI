<table class="sphinxhide">
 <tr>
   <td align="center"><img src="https://raw.githubusercontent.com/Xilinx/Image-Collateral/main/xilinx-logo.png" width="30%"/><h1>Vitis AI</h1><h0>Adaptable & Real-Time AI Inference Acceleration</h0>
   </td>
 </tr>
</table>


<br />

As of the Vitis™ AI 3.0 release, we have migrated the historic Github repository documentation to [Github.IO](https://xilinx.github.io/Vitis-AI/).

The advantages are:

- Improved quality and usefulness of documentation.
- Enhanced look and feel intended to improve the user experience.
- Developers working on air-gapped machines can use the HTML versions of the documentation directly while working offline.

The new documentation is created by consolidating source files in .rst format into the `/docsrc` directory and is used as the source repository to compile the HTML version of the documentation. The resulting HTML can be found in the `/docs` directory and can be used offline immediately once the user clones the repository. The HTML documentation in the `/docs` folder is identical to what is published on Github.IO.

If you need to rebuild the offline HTML documentation, create a sphinx-build environment in Anaconda, navigate to the `/docsrc` directory and execute the command:

```text
make github
```

Once the build process is completed, you can find the compiled HTML documentation in the `/docs` folder.

<br /> <br />
