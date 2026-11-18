"""Test free-cloud route protection and upload constraints locally."""
from fastapi.testclient import TestClient
import main


def test_cloud_access_and_single_service(monkeypatch):
    monkeypatch.setattr(main, 'CLOUD_DEPLOYMENT', True)
    monkeypatch.setenv('STUDIO_PASSWORD','test-only-unique-password')
    client=TestClient(main.app)
    assert client.get('/health').json() == {'status':'ok'}
    assert client.get('/').status_code == 401
    assert client.get('/styles.css').status_code == 401
    assert client.get('/api/jobs').status_code == 401
    credentials=('studio','test-only-unique-password')
    page = client.get('/',auth=credentials)
    assert page.status_code == 200 and b'Srujana Studio' in page.content
    assert client.get('/styles.css',auth=credentials).status_code == 200
    assert client.get('/app.js',auth=credentials).status_code == 200
    assert client.get('/api/config',auth=credentials).json()['turntable_only'] is True
    assert client.get('/api/jobs',auth=('studio','incorrect')).status_code == 401
    assert client.get('/api/jobs/../../README.md',auth=credentials).status_code in (400,404)
    assert client.post('/api/jobs',auth=credentials,
        files={'video':('capture.mp4',b'video','video/mp4')},
        data={'mode':'cpu','quality':'fast'}).status_code == 422
    assert client.post('/api/jobs',auth=credentials,
        files={'video':('capture.mp4',b'video','video/mp4')},
        data={'mode':'turntable','quality':'high'}).status_code == 422


def test_cloud_rejects_missing_password(monkeypatch):
    monkeypatch.setattr(main, 'CLOUD_DEPLOYMENT', True)
    monkeypatch.delenv('STUDIO_PASSWORD', raising=False)
    client=TestClient(main.app)
    assert client.get('/health').status_code == 200
    assert client.get('/').status_code == 503
