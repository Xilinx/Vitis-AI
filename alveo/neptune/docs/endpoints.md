# Endpoints

This document describes the current REST endpoints in Neptune.
All endpoints use the address `http://<server_hostname>:<port>` as the base.

There are a number of web pages that are already part of Neptune.
Under the hood, these pages use HTML/JS/CSS to issue a set of REST calls to the various URLs to fetch information and render it in a useful way in HTML. 
By using the REST API yourself, you can always talk to Neptune programatically, use your own clients or make your own web pages.

|Command|Endpoint|REST|Arguments|Response|
|-------|--------|----|---------|-------|
|View dashboard|`/`|GET|None|Renders [index.html](../templates/index.html)|
|Get services|`/services/list`|GET|None|A dictionary containing a list of all services available in Neptune, along with their current state (started/stopped), URL and throughput.|
|Query services|`/services/query`|GET|_service_: name of service to query|Dictionary containing detailed information about the service, including the arguments it takes, if available
|Starting a service|`/services/start`|GET, POST|_id_: name of the service to start (required)<br> _args_: dictionary containing any runtime arguments for the node (optional)| Text: "service started"|
|Stopping a service|`/services/stop`|GET|_id_: name of the service to stop (required)| Text: "service stopped"|
|Construct a service|`services/construct`|POST|A dictionary containing the recipe (from a Recipe object's `to_json()` method)|Text: "service \<name\> constructed at /serve/\<url\>"|
|Destroy a service|`/services/destruct`|POST|_url_: partial URL of the service <br> _name_: name of the service|Text: "service destroyed at /serve/\<url>"|
|Access a service| `/serve/<foo>`| Varies|Varies|The request is submitted to service _foo_ which responds as the service defines.
|Render HTML|`/render/<foo>`|GET|None|Renders `foo.html`
