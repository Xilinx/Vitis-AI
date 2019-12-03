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
import requests
import pytest

from neptune.neptune_api import NeptuneService
from neptune.tests.conftest import get_server_addr

@pytest.mark.fpgas(1)
def test_coco(request):
    """
    Check the coco service from Neptune with a known image

    Args:
        request (fixture): get the cmdline options
    """
    server_addr = get_server_addr(request.config)

    # create Neptune service and start it
    service = NeptuneService(server_addr, 'coco')
    service.start()

    # submit job to service
    post_data = {
        'url': 'http://farm1.staticflickr.com/26/50531313_4422f0787e_z.jpg',
        'dtype': 'uint8'
    }
    r = requests.post('%s/serve/coco' % server_addr, post_data)
    assert r.status_code == 200, r.text

    response = r.json()
    assert type(response) is dict

    # for this known image, validate the expected response
    for i, j in zip(response['resized_shape'], [149, 224, 3]):
        assert i == j
    assert 'img' in response
    assert response['url'] == post_data['url']
    assert len(response['boxes']) == 2
    tolerance = 5
    for i, j in zip(response['boxes'][0], [85, 18, 149, 118, "giraffe"]):
        if isinstance(j, int):
            assert j - tolerance <= i <= j + tolerance
        else:
            assert i == j
    for i, j in zip(response['boxes'][1], [21, 90, 65, 148, "zebra"]):
        if isinstance(j, int):
            assert j - tolerance <= i <= j + tolerance
        else:
            assert i == j

    service.stop()
