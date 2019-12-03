import os
import pytest
import requests

from neptune.neptune_api import NeptuneSession
import neptune.recipes.recipes as Recipes
from neptune.tests.conftest import get_server_addr

TEST_NAME = 'tmp'

# @pytest.fixture(params=['ping', pytest.param('fpga', marks=pytest.mark.fpga(1))])
@pytest.fixture(params=[
    'recipe_ping', # tests constructing a single service node
    'recipe_facedetect', # tests constructing two node services
    'recipe_coco' # tests multinode services (with last node having two outputs)
])
def new_service(request):
    recipe = getattr(Recipes, request.param)()
    recipe.name = TEST_NAME
    recipe.url = TEST_NAME
    return recipe.to_json()

@pytest.mark.timeout(10)
def test_construct(request, new_service):
    """
    Constructs a new service and makes sure it's added. Then it destroys the
    new service and makes sure it's deleted

    Args:
        request (fixture): get the cmdline options
        new_service (fixture): a Recipe object
    """

    server_addr = get_server_addr(request.config)

    session = NeptuneSession(server_addr)

    # get a list of services and make sure TEST_NAME isn't there
    response = session.get_services()

    for service in response['services']:
        assert service['name'] != TEST_NAME
        assert service['url'] != "/serve/" + TEST_NAME

    # construct the new service
    r = requests.post('%s/services/construct' % server_addr, json=new_service)
    assert r.status_code == 200, r.text

    # check if the new service exists in the service manager
    response = session.query_service(TEST_NAME)
    assert 'url' in response
    assert response['url'] == "/serve/" + TEST_NAME

    recipe_cache = os.environ['VAI_ALVEO_ROOT'] + '/neptune/recipes/recipe_%s.bak' % TEST_NAME
    assert os.path.exists(recipe_cache)

    # destroy the service
    r = requests.post('%s/services/destruct' % server_addr,
        {'url': TEST_NAME, 'name': TEST_NAME})
    assert r.status_code == 200, r.text

    # make sure the service is gone
    response = session.get_services()
    for service in response['services']:
        assert service['name'] != TEST_NAME
        assert service['url'] != "/serve/" + TEST_NAME

    assert not os.path.exists(recipe_cache)
