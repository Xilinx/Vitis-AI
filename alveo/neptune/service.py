import os

class Service(object):
    """
    The base class for all services. All services inherit from this class
    """

    def __init__(self, prefix, artifacts, graph):
        self._artifacts = artifacts
        self._prefix = prefix
        self._proc = None
        self._graph = graph


    def start(self, args):
        # os.environ["XDNN_VERBOSE"] = "1"
        # os.environ["XBLAS_EMIT_PROFILING_INFO"] = "1"
        self._graph.serve(args, background=True)


    def stop(self):
        self._graph.stop()
