import pytest

from stockbot.tools.reddit import RedditClient


class FakeResponse:
    def __init__(self, payload, status_code=200):
        self._payload = payload
        self.status_code = status_code

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


class FakeSession:
    def __init__(self):
        self.calls = []

    def post(self, url, **kwargs):
        self.calls.append(("POST", url, kwargs))
        return FakeResponse({"access_token": "not-for-output", "expires_in": 3600})

    def get(self, url, **kwargs):
        self.calls.append(("GET", url, kwargs))
        return FakeResponse({"data": {"children": []}})


def test_authenticated_listing_obtains_token_and_reuses_it():
    session = FakeSession()
    client = RedditClient(
        client_id="client", client_secret="secret", user_agent="stockbot-tests/1.0",
        session=session, clock=lambda: 100.0,
    )

    client.get("/r/stocks/new", params={"limit": 2})
    client.get("/r/stocks/search", params={"q": "AAPL"})

    assert [call[0] for call in session.calls] == ["POST", "GET", "GET"]
    token_call, first_listing = session.calls[:2]
    assert token_call[1] == "https://www.reddit.com/api/v1/access_token"
    assert token_call[2]["auth"] == ("client", "secret")
    assert token_call[2]["data"] == {"grant_type": "client_credentials"}
    assert first_listing[1] == "https://oauth.reddit.com/r/stocks/new"
    assert first_listing[2]["headers"] == {
        "Authorization": "bearer not-for-output", "User-Agent": "stockbot-tests/1.0"
    }


def test_absent_credentials_uses_degraded_public_json_fallback():
    session = FakeSession()
    client = RedditClient(client_id=None, client_secret=None, session=session)

    client.get("/r/stocks/new", params={"limit": 2})

    assert len(session.calls) == 1
    method, url, kwargs = session.calls[0]
    assert method == "GET"
    assert url == "https://www.reddit.com/r/stocks/new.json"
    assert "Authorization" not in kwargs["headers"]
    assert client.degraded_public_fallback is True


@pytest.mark.parametrize(
    ("client_id", "client_secret"), [("client", None), (None, "secret")]
)
def test_partial_credentials_are_rejected(client_id, client_secret):
    with pytest.raises(ValueError, match="both REDDIT_CLIENT_ID and REDDIT_CLIENT_SECRET"):
        RedditClient(client_id=client_id, client_secret=client_secret)


class RefreshSession(FakeSession):
    def __init__(self, get_statuses):
        super().__init__()
        self.get_statuses = iter(get_statuses)
        self.token_number = 0

    def post(self, url, **kwargs):
        self.calls.append(("POST", url, kwargs))
        self.token_number += 1
        return FakeResponse({"access_token": f"token-{self.token_number}", "expires_in": 3600})

    def get(self, url, **kwargs):
        self.calls.append(("GET", url, kwargs))
        return FakeResponse({"data": {"children": []}}, next(self.get_statuses))


def test_authenticated_401_refreshes_token_and_retries_once():
    session = RefreshSession([401, 200])
    client = RedditClient("client", "secret", session=session, clock=lambda: 100.0)

    response = client.get("/r/stocks/new")

    assert response.status_code == 200
    assert [call[0] for call in session.calls] == ["POST", "GET", "POST", "GET"]
    assert session.calls[-1][2]["headers"]["Authorization"] == "bearer token-2"


def test_second_authenticated_401_is_returned_without_infinite_retry():
    session = RefreshSession([401, 401])
    client = RedditClient("client", "secret", session=session, clock=lambda: 100.0)

    response = client.get("/r/stocks/new")

    assert response.status_code == 401
    assert [call[0] for call in session.calls] == ["POST", "GET", "POST", "GET"]
