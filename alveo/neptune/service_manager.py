# Copyright 2019 Xilinx Inc.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import concurrent.futures
import logging

from vai.dpuv1.rt import xstream

logger = logging.getLogger(__name__)

class ServiceManager(object):
    class __Impl(object):
        """ Implementation of the singleton interface """

        def __init__(self):
            self._services = {}
            self._runner = None
            self._threadpool = concurrent.futures.ThreadPoolExecutor(max_workers=10)
            self.STOPPED = -1
            self.DEAD = 0
            self.STARTING = 1
            self.STARTED = 2

        def start(self, service_name, args={}):
            if service_name not in self._services:
                logger.warning('Service %s does not exist and cannot be started' % service_name)
                return
            svc = self._services[service_name]

            if svc['state'] == self.STOPPED:
                # Start all the nodes
                svc['service'].start(args)
                svc['state'] = self.STARTING
                logger.status('starting service %s' % service_name)
            else:
                logger.warning('service %s cannot be started: state is %s' % (service_name, svc['state']))

        def stop(self, service_name):
            if service_name not in self._services:
                logger.warning('Service %s does not exist and cannot be stopped' % service_name)
                return
            svc = self._services[service_name]
            if svc['state'] >= self.DEAD:
                svc['service'].stop()
                svc['state'] = self.STOPPED
                logger.status('stopping service %s' % service_name)
            else:
                logger.warning('service %s already stopped' % service_name)

        def stop_all(self):
            for name, svc in self._services.items():
                if svc['state'] >= self.DEAD:
                    self.stop(name)

        def run(self, service_name, nparr, meta):
            if self._runner is None:
                self._runner = xstream.Runner(timeout=10000)

            (inputs, outputs) = self._get_graph_io(service_name)
            if not inputs or not outputs:
                logger.warning('Nothing returned from service %s' % service_name)
                return None

            # If callback is in the meta, it's streaming so no return is expected
            # Note, this might be masking an issue. If we don't have this, in
            # streaming, reopening a render/xdf with certain streams (e.g. coco
            # and facedetect), results in a hang because the run function doesn't
            # return.
            if 'callback_id' in meta:
                self._runner.send(inputs[0], nparr, meta)
                return None

            return self._runner.run(inputs[0], outputs[0], nparr, meta)

        def send(self, service_name, index, nparr, meta):
            if self._runner is None:
                self._runner = xstream.Runner(timeout=-1)

            (inputs, outputs) = self._get_graph_io(service_name)
            if not inputs or not outputs:
                logger.warning('Nothing returned from service %s' % service_name)
                return None

            start_channel = inputs[index]

            self._runner.send(start_channel, nparr, meta)

        def list(self):
            return self._services

        def _get_graph_io(self, service_name):
            if (service_name not in self._services) or self._services[service_name]['state'] < self.DEAD:
                logger.warning('Service %s not found or started' % service_name)
                return ([], [])

            service = self._services[service_name]['service']
            return (service._graph._in, service._graph._out)

        def add(self, service_name, service, service_url, args):
            if service_name in self._services:
                logger.warning('Service %s cannot be added, already exists' % service_name)
                return
            self._services[service_name] = {
                'service': service,
                'state': self.STOPPED,
                'url': service_url,
                'throughput': {},
                'args': args
            }
            logger.status('adding service %s' % service_name)

        def remove(self, service_name):
            if service_name not in self._services:
                logger.warning('Service %s cannot be deleted, as it does not exist' % service_name)
                return
            del self._services[service_name]
            logger.status('deleting service %s' % service_name)

        def update_throughput_stats(self, service_name, edge_name, throughput):
            if service_name not in self._services:
                return
            svc = self._services[service_name]
            if 'throughput' not in svc:
                svc['throughput'] = {}

            svc['throughput'][edge_name] = throughput

    # storage for the instance reference
    __instance = None

    def __init__(self):
        """ Create singleton instance """
        # Check whether we already have an instance
        if ServiceManager.__instance is None:
            # Create and remember instance
            ServiceManager.__instance = ServiceManager.__Impl()

        # Store instance reference as the only member in the handle
        self.__dict__['_Service_Manager__instance'] = ServiceManager.__instance

    def __getattr__(self, attr):
        """ Delegate access to implementation """
        return getattr(self.__instance, attr)

    def __setattr__(self, attr, value):
        """ Delegate access to implementation """
        return setattr(self.__instance, attr, value)

    def _drop(self):
        "Drop the instance (for testing purposes)."
        ServiceManager.__instance = None
        del self._Service_Manager__instance
