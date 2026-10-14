import os

from fastapi.testclient import TestClient

# Force fixture source for API tests.
os.environ["ARBITRA_TOOLS"] = "mock"

from src.api.app import app  # noqa: E402


client = TestClient(app)


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True


def test_list_cases_and_run_stub_case():
    cases = client.get("/cases")
    assert cases.status_code == 200
    ids = cases.json()["cases"]
    assert ids

    did = ids[0]
    out = client.post(f"/cases/{did}/run", json={"config": "C", "stub": True})
    assert out.status_code == 200
    body = out.json()
    assert body["dispute_id"] == did
    assert body["verdict"] is not None
    assert "ruling" in body["verdict"]


def test_eval_endpoint_stub_fixtures():
    out = client.post("/eval", json={"config": "C", "split": "dev", "source": "fixtures", "limit": 2})
    assert out.status_code == 200
    body = out.json()
    assert body["n"] == 2
    assert "summary" in body
    assert body["summary"]["n"] == 2
