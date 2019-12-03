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
"""
Neptune is a Python web server for using FPGAs in the cloud. This API can
be used by Neptune clients to interact with the server.
"""

import time

import requests
from requests.exceptions import ConnectionError

from neptune.constants import SERVICE_STARTED, SERVICE_STOPPED


class NeptuneSession(object):
    """
    Start a Neptune session at a given address

    Raises:
        ConnectionError: On failure to connect to the server

    TODO: fold NeptuneService into the session
    """
    def __init__(self, server_addr):
        """
        Start a Neptune session at a given address

        Args:
            server_addr (str): e.g. "http://<hostname>:<port>"
        """
        self.addr = server_addr

    def get_services(self):
        """
        Gets a dict of the services available on Neptune

        Raises:
            ConnectionError: On failure to connect to Neptune

        Returns:
            dict: Response from Neptune
        """
        try:
            r = requests.get('%s/services/list' % self.addr)
        except ConnectionError as e:
            raise ConnectionError("Failed to connect to %s" % self.addr)
        assert r.status_code == 200, r.text

        response = r.json()
        assert type(response) is dict

        return response

    def query_service(self, name):
        """
        Query detailed information on a specific service. This contains information
        not found in get_services()

        Args:
            name (str): Name of the service to query

        Raises:
            ConnectionError: On failure to connect to Neptune

        Returns:
            dict: Response from Neptune
        """
        try:
            r = requests.get('%s/services/query?service=%s' % (self.addr, name))
        except ConnectionError as e:
            raise ConnectionError("Failed to connect to %s" % self.addr)
        assert r.status_code == 200, r.text

        response = r.json()
        assert type(response) is dict

        return response

class NeptuneService(object):
    """
    Used to interact with a Neptune service
    """

    def __init__(self, server_addr, service_name):
        """
        Used to interact with a Neptune service

        Args:
            server_addr (str): Address of the Neptune server
            service_name (str): Name of the service
        """
        self.addr = server_addr
        self.name = service_name

    def start(self, args=None):
        """
        Start the service

        Args:
            args (NeptuneServiceArgs, optional): Specify any runtime args. Defaults to None.

        Raises:
            ConnectionError: On failure to connect to Neptune
        """
        data = {
            'id': self.name
        }
        if args is not None:
            data['args'] = args.args
        try:
            r = requests.post(
                '%s/services/start' % (self.addr),
                json=data
            )
        except ConnectionError as e:
            raise ConnectionError("Failed to connect to %s" % self.addr)
        assert r.status_code == 200

        self._wait_until_started()

    def _wait_until_started(self):
        """
        Wait until the service is started before continuing
        """
        started = False
        while not started:
            r = requests.get('%s/services/list' % self.addr)
            assert r.status_code == 200, r.text

            response = r.json()
            assert type(response) is dict

            for service in response['services']:
                if service['name'] == self.name:
                    if service['state'] == SERVICE_STARTED:
                        started = True

            time.sleep(1)

    def _wait_until_stopped(self):
        """
        Wait until the service is stopped before continuing
        """
        stopped = False
        while not stopped:
            r = requests.get('%s/services/list' % self.addr)
            assert r.status_code == 200, r.text

            response = r.json()
            assert type(response) is dict

            for service in response['services']:
                if service['name'] == self.name:
                    if service['state'] == SERVICE_STOPPED:
                        stopped = True

            time.sleep(1)

    def stop(self):
        """
        Stop the service
        """
        r = requests.get(
            '%s/services/stop?id=%s' % (self.addr, self.name)
        )
        assert r.status_code == 200
        self._wait_until_stopped()

class NeptuneServiceArgs(object):
    """
    A thin wrapper around a dictionary for arguments for Neptune.

    {
        node_0: {
            add: {
                key_0: value_0,
                key_1: value_1
            },
            remove: {
                ...
            }
        }
        ...
    }
    Add: add new arguments or update any specified in the recipe
    Remove: remove arguments from the recipe arguments
    """
    def __init__(self):
        """
        A thin wrapper around a dictionary for arguments for Neptune.
        """
        self.args = {}

    def insert_add_arg(self, node, key, value):
        """
        Add a new argument for a node using a key/value pair

        Args:
            node (str): Node to add the argument to
            key (str): Key of the argument
            value (str): Value of the argument
        """
        if node not in self.args:
            self.args[node] = {}
        if 'add' not in self.args[node]:
            self.args[node]['add'] = {}
        self.args[node]['add'][key] = value

    def insert_remove_arg(self, node, key, value):
        """
        Add an argument for a node using a key/value pair to be removed
        from the default arguments in the recipe

        Args:
            node (str): Node to remove the argument from
            key (str): Key of the argument
            value (str): Value of the argument
        """
        if node not in self.args:
            self.args[node] = {}
        if 'remove' not in self.args[node]:
            self.args[node]['remove'] = {}
        self.args[node]['remove'][key] = value

    def delete_add_arg(self, node, key):
        """
        Delete an argument from the NeptuneServiceArgs object

        Args:
            node (str): Node to remove argument from
            key (str): Key of the argument to delete
        """
        if key in self.args[node]['add']:
            del self.args[node]['add'][key]

    def delete_remove_arg(self, node, key):
        """
        Delete an argument from the NeptuneServiceArgs object

        Args:
            node (str): Node to remove argument from
            key (str): Key of the argument to delete
        """
        if key in self.args[node]['remove']:
            del self.args[node]['remove'][key]

    def get_add_arg(self, node, key):
        """
        Get the value of an argument from the NeptuneServiceArgs object

        Args:
            node (str): Node which has the argument
            key (str): Key of the argument to get
        """
        if key in self.args[node]['add']:
            return self.args[node]['add'][key]
        return None

    def get_remove_arg(self, node, key):
        """
        Get the value of an argument from the NeptuneServiceArgs object

        Args:
            node (str): Node which has the argument
            key (str): Key of the argument to get
        """
        if key in self.args[node]['remove']:
            return self.args[node]['remove'][key]
        return None

    def __str__(self):
        retval = ""
        for nodes, args in self.args.items():
            retval += nodes + "\n"
            if 'add' in args:
                retval += "  add\n"
                for key, value in args['add'].items():
                    retval += "    %s: %s\n" % (key, str(value))
            if 'remove' in args:
                retval += "  remove\n"
                for key, value in args['remove'].items():
                    retval += "    %s: %s\n" % (key, str(value))
        return retval.rstrip()
