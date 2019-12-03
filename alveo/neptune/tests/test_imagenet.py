import requests
import pytest

from neptune.neptune_api import NeptuneService
from neptune.tests.conftest import get_server_addr

@pytest.mark.fpgas(1)
def test_imagenet(request):
    """
    Check the imagenet service from Neptune with a known image

    Args:
        request (fixture): get the cmdline options
    """
    server_addr = get_server_addr(request.config)

    # create Neptune service and start it
    service = NeptuneService(server_addr, 'imagenet')
    service.start()

    # submit job to service
    post_data = {
        'url': 'https://www-tc.pbs.org/wnet/nature/files/2018/07/Bear133-1280x720.jpg',
        'dtype': 'uint8'
    }
    r = requests.post('%s/serve/imagenet' % server_addr, post_data)
    assert r.status_code == 200, r.text

    response = r.json()
    assert type(response) is dict

    assert 'img' in response
    assert response['url'] == post_data['url']
    assert "predictions" in response
    assert "boxes" in response
    predictions = response["predictions"]

    #TODO verify output response

    service.stop()
