import io
import urllib.error
from email.message import Message
from unittest.mock import Mock

import httpx
import pytest

from molquarry.http import HttpClient
from molquarry.transport import StdlibTransport


def response(code=200, content=b'{"studies":[]}'):
    headers = Message()
    headers["Content-Type"] = "application/json"
    return urllib.error.HTTPError(
        "https://clinicaltrials.gov/api/v2/studies", code, "status", headers, io.BytesIO(content)
    )


def test_native_clinical_transport_routing_and_response():
    with_client = HttpClient()
    try:
        native = with_client.client._transport_for_url(httpx.URL("https://clinicaltrials.gov"))
        assert isinstance(native, StdlibTransport)
        assert not isinstance(
            with_client.client._transport_for_url(httpx.URL("https://www.ebi.ac.uk")),
            StdlibTransport,
        )
        native.opener = Mock()
        native.opener.open.return_value = response()
        result = with_client.client.get("https://clinicaltrials.gov/api/v2/studies?query.term=EGFR")
        assert result.json() == {"studies": []}
        req = native.opener.open.call_args.args[0]
        assert req.full_url.endswith("query.term=EGFR")
        assert req.get_header("Accept-encoding") == "identity"
    finally:
        with_client.close()


def test_native_transport_preserves_403_and_timeout():
    transport = StdlibTransport()
    transport.opener = Mock()
    request = httpx.Request("GET", "https://clinicaltrials.gov/api/v2/studies")
    transport.opener.open.side_effect = response(403, b"Forbidden")
    result = transport.handle_request(request)
    assert result.status_code == 403 and result.content == b"Forbidden"
    assert transport.opener.open.call_count == 1
    transport.opener.open.side_effect = urllib.error.URLError(TimeoutError())
    with pytest.raises(httpx.ReadTimeout):
        transport.handle_request(request)
