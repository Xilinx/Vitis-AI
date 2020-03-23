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
import os
import pytest
import subprocess
import time
import socket
import signal
import py
import sys
import requests

neptune_command = []
server_addr = ""
root_path = os.environ['VAI_ALVEO_ROOT']
run_path = root_path + '/neptune/run.sh'

# The timeout code is taken from https://stackoverflow.com/a/46773292
class Termination(SystemExit):
    pass


class TimeoutExit(BaseException):
    pass


def _terminate(signum, frame):
    raise Termination("Runner is terminated from outside.")

def isUp(server_addr):
    try:
        requests.get('%s/services/list' % server_addr)
    except:
        time.sleep(2)
        return False
    else:
        return True


def _timeout(signum, frame):
    global neptune_command
    global server_addr

    if neptune_command:
        print("\nRestarting Neptune")
        subprocess.call(["pkill", "-9", "-f", "server.py"])
        p = subprocess.Popen(neptune_command)
        while not isUp(server_addr):
            pass

    # for some reason, without trying to print (which raises an exception),
    # the timeout isn't raised and the test succeeds despite timing out
    # Update: different behavior seen on another machine but left here in case
    try:
        print("Raising timeout exception")
    except KeyboardInterrupt:
        print("Raised keyboard exception")
    raise TimeoutExit()

@pytest.hookimpl
def pytest_sessionstart(session):
    global neptune_command
    global server_addr
    proxyEnvs = ['HTTP_PROXY', 'http_proxy', 'HTTPS_PROXY', 'https_proxy']
    for p in proxyEnvs:
      if p in os.environ:
        del os.environ[p]

    print("\nStarting Neptune")
    mode = session.config.getoption("--neptune_mode")
    if mode != 'n':
        neptune_command = ['bash', run_path]
        coverage = session.config.getoption("--coverage")
        port = session.config.getoption("--port")
        wsport = session.config.getoption("--wsport")

        if mode == "q":
            neptune_command.append("--quiet")
        elif mode == "c":
            neptune_command.append("--clean")
        elif mode == "d":
            pass
        else:
            neptune_command.append("--quiet")
            neptune_command.append("--clean")
        if coverage:
            neptune_command.append("--cov")
        neptune_command.extend(["--port", port, "--wsport", wsport])
        p = subprocess.Popen(neptune_command)
    server_addr = get_server_addr(session.config)
    while not isUp(server_addr):
        pass

@pytest.hookimpl
def pytest_sessionfinish(session, exitstatus):
    global neptune_command
    global server_addr

    if neptune_command:
        print("\nEnding Neptune")
        subprocess.call(["pkill", "-2", "-f", "server.py"])
        while isUp(server_addr):
            pass

@pytest.hookimpl
def pytest_addoption(parser):
    parser.addoption("--hostname", action="store", default="localhost")
    parser.addoption("--port", action="store", default=8998)
    parser.addoption("--wsport", action="store", default=8999)
    parser.addoption("--fpgas", action="store", default=0)
    parser.addoption("--benchmark", action="store", default="skip")
    parser.addoption("--neptune_mode", action="store", default="qc")
    parser.addoption("--coverage", action="store_true", default=False)

def get_server_addr(config):
    """
    Get the address of the Neptune server

    Args:
        config (Config): Pytest config object containing the options

    Returns:
        str: Address of the Neptune server
    """
    hostname = config.getoption('hostname')
    port = config.getoption('port')
    return "http://%s:%s" % (hostname, str(port))

def get_ws_addr(config):
    """
    Get the address of the Neptune's websocket server

    Args:
        config (Config): Pytest config object containing the options

    Returns:
        str: Address of Neptune's websocket server
    """
    hostname = config.getoption('hostname')
    port = config.getoption('wsport')
    return "ws://%s:%s" % (hostname, str(port))

@pytest.hookimpl
def pytest_configure(config):
    # register an additional marker
    config.addinivalue_line(
        "markers", "timeout(time): add a timeout value for this test"
    )

    config.addinivalue_line(
        "markers", "fpgas(number): indicate minimum number of FPGAs needed for this test"
    )

    config.addinivalue_line(
        "markers", "benchmark: indicates a benchmark test with logging output"
    )

    # Install the signal handlers that we want to process.
    signal.signal(signal.SIGTERM, _terminate)
    signal.signal(signal.SIGALRM, _timeout)

@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_protocol(item, nextitem):
    # putting this in runtest_setup or call suppresses the print until the end
    tw = py.io.TerminalWriter(sys.stdout)
    tw.line()
    tw.sep("=", "starting test", bold=True)

    marker = item.get_closest_marker('timeout')
    if marker:
        signal.alarm(marker.args[0])

    try:
        # Run the setup, test body, and teardown stages.
        yield
    finally:
        # Disable the alarm when the test passes or fails.
        # I.e. when we get into the framework's body.
        signal.alarm(0)

def pytest_collection_modifyitems(config, items):
    benchmark_mode = config.getoption("--benchmark")
    skip_bench = pytest.mark.skip(reason="use --benchmark option to run")
    if benchmark_mode == "only":
        for item in items:
            if "benchmark" not in item.keywords:
                item.add_marker(skip_bench)
    elif benchmark_mode == "all":
        pass
    else:
        for item in items:
            if "benchmark" in item.keywords:
                item.add_marker(skip_bench)

    fpgas_avail = int(config.getoption("--fpgas"))
    for item in items:
        if "fpgas" in item.keywords:
            fpgas_req = int(item.get_closest_marker('fpgas').args[0])
            if fpgas_req > fpgas_avail:
                skip_fpga = pytest.mark.skip(
                    reason="Needs %d FPGA(s). Use --fpgas option to specify" % fpgas_req
                )
                item.add_marker(skip_fpga)
