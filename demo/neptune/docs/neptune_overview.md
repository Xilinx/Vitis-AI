# Neptune Overview

This document provides an overall look at how Neptune works and goes over some of the terminology.

---
## Table of Contents <!-- omit in toc -->
- [What is it?](#what-is-it)
- [Services](#services)
- [Nodes](#nodes)
- [Using Neptune](#using-neptune)

---

## What is it?

Neptune is a Python web server built with [Tornado](https://www.tornadoweb.org/en/stable/). 
Neptune defines several REST [endpoints](endpoints.md) that clients can interact with.
For example, there's a URL that renders the Neptune dashboard where users can start and stop [services](#services).
Services perform useful work upon request from a client.
They enable the ability to leverage Vitis-AI and FPGAs in running machine learning inference.

## Services

Services form the core feature of Neptune. 
Services enable access to what the user wants to do.

For example, one simple service that Neptune has is the _ping_ service.
It responds to GET and POST requests and simply replies back with the current time. 
A more complex service is _sface_.
This service requires a URL to a video (among some other information).
It uses the video at the URL and reads frames from it at the requested FPS.
Face-detection is run on these frames (using a precompiled model) and the frame (along with coordinates for any faces found) are sent back to the client through a websocket.
The client can then use the information as required.
This service is part of three shown in the [demo picture](docs/neptune_demo.png).

A service consists of several components: a name, a URL, a dataflow graph and handler(s).
* **Name** and **URL**: they define the name of the service and what URL the service should be accessed through.
While both the name and URL are manually specified for now, for most purposes, they should be the same to ensure uniqueness.
In the future, these two may be merged to be a single value for both.
* **Graph**: The dataflow graph defines what work a service should do. 
It is made up of one or more [nodes](#nodes) that are chained together in some specified way.
Each node operates as a separate process and uses pub/sub to communicate data between them.
* **Handler**: A handler governs how clients communicate to the service.
It is responsible for responding to REST requests, taking any arguments provided, and passing them to the service to start it. 
After the service finishes, it should also reply back to the client with whatever data may be appropriate.

Refer to the [service documentation](../recipes/README.md) for more information about how to write your own service.

## Nodes

A node is used to do work in Neptune.
It can be as simple or complex as needed.
Since each node in a service operates as a separate process, breaking a single complex node into several may improve throughput of the service.
When adding a node to a service, it can take a dictionary of arguments.
Through this argument dictionary, the same node can operate differently in different services, enabling reuse.

The base class (located at vai/dpuv1/rt/xsnodes/base.py) for nodes in Neptune shows the structure that nodes should use.
For more complex nodes, the default methods can also be overridden.

Refer to the node documentation (located at /vai/dpuv1/rt/xsnodes/README.md) for more information about how to write your own nodes.

## Using Neptune

At its core, using Neptune is as simple as:
1. Start Neptune
2. Start the service(s) of interest
3. Send it requests

Out of the box, Neptune comes with the tools to perform these three tasks with some default services.
The address `http://<server_hostname>:<port>/` shows a dashboard that can be used to start and stop services.
It can also be used to send some basic requests to services through the demo section.
In addition, Neptune comes with some HTML/JS/CSS to issue requests and render responses from services.
Of course, these tools are just wrappers around the basic REST endpoints that Neptune exposes.
From the [endpoints documentation](endpoints.md) or code, you can see what the endpoint addresses are (or how we talk to them) and then talk to them yourself.
