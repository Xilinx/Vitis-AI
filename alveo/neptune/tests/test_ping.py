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
import logging
import pytest
import requests
import threading
import time

from neptune.neptune_api import NeptuneService, NeptuneServiceArgs
from neptune.tests.conftest import get_server_addr

@pytest.mark.timeout(15)
def test_ping(request):
    """
    Gets n pings from Neptune using the ping service

    Args:
        request (fixture): get the cmdline options
    """
    server_addr = get_server_addr(request.config)
    service = NeptuneService(server_addr, 'ping')

    service.start()

    r = requests.get('%s/serve/ping' % server_addr)
    assert r.status_code == 200

    response = r.json()
    # verify both the id and pong keys are present in the response
    assert type(response) is dict
    assert 'pong' in response
    assert 'id' in response

    float(response['pong']) # check if pong is a float

    service.stop()

@pytest.fixture(params=[
    0, # don't spawn processes
    1,
    8
])
def args_ping(request):
    if request.param is None:
        return None
    else:
        args = NeptuneServiceArgs()
        args.insert_add_arg('ping', 'processes', request.param)
        return args

def _run_test_ping_benchmark(server_addr, iterations):
    for i in range(iterations):
        r = requests.get('%s/serve/ping' % server_addr)
        assert r.status_code == 200

        response = r.json()

        # verify both the id and pong keys are present in the response
        assert type(response) is dict
        assert 'pong' in response
        assert 'id' in response

        float(response['pong']) # check if pong is a float

        # optional delay between pings
        # time.sleep(0.1)

@pytest.mark.timeout(30)
@pytest.mark.benchmark
def test_ping_benchmark(request, args_ping):
    """
    Benchmarks ping times using args_ping to configure the ping service

    Args:
        request (fixture): get the cmdline options
        args_ping (NeptuneServiceArgs): a Neptune service args object
    """
    server_addr = get_server_addr(request.config)
    service = NeptuneService(server_addr, 'ping')
    ITERATION_COUNT = 1
    THREAD_COUNT = 8

    service.start(args_ping)
    processes_count = args_ping.get_add_arg('ping', 'processes')
    threads = []

    for i in range(THREAD_COUNT):
        thread = threading.Thread(
            target=_run_test_ping_benchmark,
            args=(server_addr, ITERATION_COUNT)
        )
        threads.append(thread)

    start_time = time.time()
    for thread in threads:
        thread.start()

    for thread in threads:
        thread.join()

    end_time = time.time()

    logging.info("ping_benchmark: %d pings took %fs with %d processes" % \
        (ITERATION_COUNT*THREAD_COUNT, (end_time - start_time), processes_count))

    service.stop()
