# Services

This document describes how to add and manipulate services in Neptune.

Services can be added to Neptune through recipes, which define what the service does and how to talk to it.
The easiest way to do this is through the [_recipe_ module](recipe.py), which provides a set of classes to simplify creating recipes.
Examples of how to use this module can be seen in the [default recipes](recipes.py) included in Neptune.

---
## Table of Contents <!-- omit in toc -->
- [Overview](#overview)
- [Adding Recipes (i.e. Services)](#adding-recipes-ie-services)
- [Caveats](#caveats)
  - [Order Matters in the Graph](#order-matters-in-the-graph)
  - [Name Length](#name-length)

---

## Overview

At a high level, a service consists of a URL, name, graph, and handler(s).
Through the helper [_recipe_ module](recipe.py), we can construct a graph, add nodes/edges to it, list handlers and aggregate this data into a _Recipe_ object.

On startup, the Neptune server will automatically use any function in this folder that starts with `recipe_` as a recipe and expect that it returns a constructed _Recipe_ object.
Using these _Recipe_ objects, Neptune creates a dictionary representation of it that's written to this folder using the naming convention of `recipe_<name>.bak`.
For future launches of the server, it will only use any `recipe_*.bak` files that may be found here.
To construct the recipes from the functions again, delete all `recipe_*.bak` files or launch Neptune with the `--clean` flag in the `run.sh` script.
Services can also be destroyed to prevent access.
Destroyed services remain destroyed until the `recipe_*.bak` cache is cleared, which will add back all the default services in Neptune when it constructs the services from the recipe functions again.

This behavior to use the dictionary representations of the recipes rather than the functions every time serves two purposes.
First, it allows you, as the user, to remove any of the default services without needing to comment or delete code.
Then, you will never see that service again as long as at least one `recipe_*.bak` file exists in this folder.
Second, it allows dynamically constructed services to persist across server launches.

Just as you can add default services to Neptune by using the _recipe_ module, clients can also construct services through the same mechanism.
This dynamic construction of services can be done live while Neptune is running.
To do so, clients can construct the _Recipe_ object on their machine and then POST the recipe as a JSON to Neptune.
Neptune will add this recipe as a service, allowing it to persist across server launches until the `recipe_*.bak` cache is cleared. The [test_construct](../tests/test_construct.py) test demonstrates an example of how to do this.

Refer to the [endpoints documentation](../docs/endpoints.md) or code to see where the REST URL endpoints are to perform construction and destruction of services.

## Adding Recipes (i.e. Services)

As mentioned above, recipes consist of a graph, a name, a URL, and handlers.
* **Graph**: the graph consists of nodes and edges.
More information about both nodes and edges can be seen in the node documentation.
  * **Nodes** are pulled from the xsnodes (located at vai/dpuv1/rt/xsnodes/) folder and can be added here by name (i.e. the file name).
Nodes use pub/sub to communicate to other nodes.
Each node can accept data from one or more nodes ([order matters](#order-matters-in-the-graph)!) and publish to a channel that multiple nodes may be listening on.
Nodes also accept arguments in the form of a dictionary that are used during operation.
At service start time, you can also pass in another argument dictionary to change the nodes' arguments from the defaults provided by the service definition.
Refer to the node documentation (located at vai/dpuv1/rt/xsnodes/README.md) for more information.
  * **Edges** are uni-directional connections between nodes and need to be added ([order matters](#order-matters-in-the-graph) here too!).
They indicate the source and sink nodes in the connection.
The implicit start/end edges to the first node and from the last node are added automatically by the _Recipe_ object.
* The **name** and **URL** are just strings.
For now, the safest thing is to set them both to the same string.
There may be some hidden dependencies that break if they're not the same.
Don't make them [too long](#name-length)!
* **Handlers** are functions that get called when the user interacts with the service.
A recipe should specify a list of handler functions for each REST call that the service should respond to.
The syntax for handlers is to provide the module name in `handlers/` and then the name of the handler.
For example, if the handler function is called `foo` in `bar.py`, this handler would be referenced with `bar.foo`.
This mechanism will be updated in the future to bring it in line with how recipes are auto-discovered so only the handler name will need to be provided.
  * Currently, GET, POST and PATCH requests are supported and a service can have different handlers for each.
  * The list of handlers is used to allow handlers to be chained together and called in succession.
  * If only a single handler function is used, it can be just a list containing a single element or just a string itself.
  * More information about handlers can be seen in its [documentation](TBD).

The easiest way to add new services is to look at existing ones and use them as a starting point.
Recipes don't need to exist solely within `recipes.py`.
Any Python file in the this folder containing a function whose name starts with `recipe_` will be used so you can group recipes together in some logical way if you'd like.
_recipe\_ping_ is an easy one to see as it's a single node service.
For a more complex service, take a look at _recipe\_coco_.

Look at the documentation [in code](recipe.py) for how to use the _Recipe_ module if unclear from its usage in the default recipes.

If required, you can also add services by writing a `*.bak` file yourself.
Take a look at the `.bak` that Neptune writes from the recipe function and then edit as needed.
However, this is not recommended because the syntax of the dictionary may change and there can be subtle gotchas.

## Caveats

A few things to keep in mind when writing services.

### Order Matters in the Graph

Nodes and edges are added in order and the service will not work well if its nodes/edges are added in a bad order.

The simple services may use a linear connection order by adding node_0 and node_1 for example.
In this case, node_0 starts the service and publishes to node_1, which publishes to end the service.
So here, node_0 should be added to the graph first and then followed by node_1.
Additionally, just a single edge between node_0 (source) and node_1 (sink) needs to be added.

To use a more complex service as an example, consider _recipe\_coco_.
It adds the following nodes: _image\_frontend_, _pre_, _fpga_, and _post_.
Edges need to be added in between all the nodes since data flows linearly through them in that order.
However, there is an additional edge required connecting _post_ (source) to _fpga_ (sink), which is used for controlling throughput in _fpga_.
The _fpga_ node, since there are two nodes sinking to it (_pre_ and _post_), uses indexing to determine which channel to listen on.
In this case, since the _pre_ -> _fpga_ edge is added first, index 0 corresponds to this connection and index 1 corresponds to the _post_ -> _fpga_ connection.
The node can get from either channel by using the appropriate index.

### Name Length

There are some restrictions on the name length.
Specifically, the inter-process backend is currently one of Plasma or Redis.
Plasma enforces a restriction that any IDs used to transmit information must be less than 20 characters.
This limitation, and the fact that the node names are used in the IDs for communication, places a limit that the service name should be less than 12 characters to keep the overall ID under 20 characters.
