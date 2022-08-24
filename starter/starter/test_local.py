"""
Unit tests for the project
"""
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)


def test_base_path():
    """ Tests the base Code path
    """
    r = client.get("/")
    assert r.status_code == 200
    assert r.json() == {"greeting": "Welcome to my MLOps project!"}


def test_get_path_query():
    """ Tests to get the correct info after Query
    """
    r = client.get("/items/42?count=12")
    assert r.status_code == 200
    assert r.json() == {'fetch': 'Fetched 12 of 42'}


def test_get_malformed():
    """ Tests malformed endpoint
    """
    r = client.get('/items')
    assert r.status_code != 200
