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
import json
import pytest
import requests

try:
    from websocket import create_connection
except ImportError:
    pytest.skip("websocket-client is required for this test", allow_module_level=True)

from neptune.neptune_api import NeptuneService
from neptune.tests.conftest import get_server_addr, get_ws_addr

@pytest.mark.fpgas(1)
def test_sface(request):
    """
    Check the streaming facedetect service from Neptune with a known video

    Args:
        request (fixture): get the cmdline options
    """
    server_addr = get_server_addr(request.config)
    ws_addr = get_ws_addr(request.config)

    service = NeptuneService(server_addr, 'sface')

    service.start()

    ws = create_connection(ws_addr)
    response = json.loads(ws.recv())
    assert isinstance(response, dict)
    assert response['topic'] == 'id'
    assert 'message' in response

    client_id = response['message']

    post_data = {
        'url': 'https://www.youtube.com/watch?v=mH9pDONwq3I',
        'dtype': 'uint8',
        'callback_id': client_id
    }

    r = requests.post('%s/serve/sface' % server_addr, post_data)
    assert r.status_code == 200, r.text

    response = r.json()
    assert type(response) is dict
    # TODO should this response be checked?

    # collect some number of frames
    foundBoxes = False
    for i in range(20):
        for j in range(10):
            response = json.loads(ws.recv())
            assert isinstance(response, dict)
            assert response['topic'] == 'callback'
            assert 'message' in response

            response = json.loads(response['message'])
            assert isinstance(response, dict)
            assert 'img' in response
            assert 'boxes' in response
            assert response['callback_id'] == client_id

            # in this video, there should be a face (i.e. boxes) in some frames
            if response['boxes']:
              foundBoxes = True

        # issue keepalive request every so often
        r = requests.post('%s/serve/sface' % server_addr, post_data)
        assert r.status_code == 200, r.text

    assert foundBoxes

    ws.close()
    service.stop()
