# Xstream - lightweight multi-process streaming system for ML+x applications

## Overview

Xstream (`vai/dpuv1/rt/xstream.py`) provides a standard mechanism for streaming data between multiple processes and controlling execution flow / dependencies.


## Xstream Node

Each Xstream Node is a stream processor. It is a separate process that can subscribe to zero or more input Channels, and output to zero or more output Channels. A Node may perform computation on Payload received on its input Channel(s). The computation can be implemented in CPU, FPGA or GPU. 


## Xstream Channel

Channels are defined by an alphanumeric string. Xstream Nodes may publish Payloads to Channels and subscribe to Channels to receive Payloads. 
The default pattern is PUB-SUB, i.e., all subscribers of a Channel will receive all Payloads published to that Channel. Payloads are queued up on the Subscriber side in FIFO order until the Subscriber consumes them off the queue.


## Xstream Payload

Xstream Payloads contain two items: a blob of binary data and metadata. 
The binary blob and metadata are transmitted using Redis as an object store. 
The binary blob is meant for large data. The metadata is meant for smaller data like IDs, arguments and options.
The object IDs are transmitted via ZMQ. ZMQ is used for stream flow control.
The `id` field is required in the metadata.
An empty Payload is used to signal "end of transmission".


## Defining a new Node

Add a new Python file in `vai/dpuv1/rt/xsnodes`. See `ping.py` as an example. Every Node should loop forever upon construction. On each iteration of the loop, it should consume Payloads from its input Channel(s) and publish Payloads to its output Channel(s). 
If an empty Payload is received, the Node should forward the empty Payload to its output Channels by calling `xstream.end()` and exit.


## Connecting Nodes to form a service Graph

Use `vai/dpuv1/rt/xsnodes/grapher.py` to construct a Graph consisting of one or more Nodes. When Graph.serve() is called, the Graph will spawn each Node as a separate Process and connect their input/output Channels. The Graph will manage the life and death of all its Nodes.
See `neptune/services/ping.py` for a Graph example.

Example:
```
  graph = grapher.Graph("my_graph")
  graph.node("prep", pre.ImagenetPreProcess, args)
  graph.node("fpga", fpga.FpgaProcess, args)
  graph.node("post", post.ImagenetPostProcess, args)

  graph.edge("START", None, "prep")
  graph.edge("fpga", "prep", "fpga")
  graph.edge("post", "fpga", "post")
  graph.edge("DONE", "post", None)

  graph.serve(background=True)
  ...
  graph.stop()

```


## Xstream Runner

The Runner is a convenience class that pushes a Payload to the input Channel of a Graph. The Payload is submitted with a unique `id`. The Runner then waits for the output Payload of the Graph matching the submitted `id`. 

The purpose of this Runner is to provide the look-and-feel of a blocking "function call".
