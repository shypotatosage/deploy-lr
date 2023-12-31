import sys
import os

# Add the root directory to the Python path
sys.path.append("../")

from deploy_lr_project.app import app


@pytest.fixture
def client():
    return app.test_client()

def test_predict_route(client):
    response = client.get('/predict/14/20/0/11/2/1/2/4/2/3/3/3/2/3/2/3/3/3/3/2')
    assert response.status_code == 200
    assert response.get_json() == 1  # Assuming your prediction is a stress level

# Add more tests for different routes and edge cases as needed
