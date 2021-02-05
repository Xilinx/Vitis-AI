<table class="sphinxhide">
 <tr>
   <td align="center"><img src="https://www.xilinx.com/content/dam/xilinx/imgs/press/media-kits/corporate/xilinx-logo.png" width="30%"/><h1>Vitis AI</h1>
   </td>
 </tr>
</table>

# Installation

Two options are available for installing the containers with the Vitis AI tools and resources.

 - Pre-built containers on Docker Hub: [xilinx/vitis-ai](https://hub.docker.com/r/xilinx/vitis-ai/tags)
 - Build containers locally with Docker recipes: [Docker Recipes](https://gitenterprise.xilinx.com/Vitis/vitis-ai-staging/tree/1.3.1/setup/docker)

 - [Install Docker](install_docker/README.md) - if Docker not installed on your machine yet

 - [Ensure your linux user is in the group docker](https://docs.docker.com/install/linux/linux-postinstall/)

 - Clone the Vitis-AI repository to obtain the examples, reference code, and scripts.
    ```bash
    git clone --recurse-submodules https://github.com/Xilinx/Vitis-AI  

    cd Vitis-AI
    ```

Refer to the following pages for installation information:

## Build from Recipe

See [Build from Recipe](install-recipe.md).

## Build from Source

See [Build from Source](install-source.md).

## Cloud Installation

See [Cloud Installation](install-cloud.md).

## Install Docker Images

See [Install Docker Images](install_docker/README.md).

## Pre-Built Image Installation

See [Pre-Built Image Installation](install-prebuilt.md).