import json

import httpx
import pytest

from molquarry import MolQuarryError
from molquarry.catalog import SOURCES
from molquarry.http import retry_after_seconds


def test_retry_then_success_honors_retry_after(make_quarry):
    sleeps, calls = [], []

    def handler(request):
        calls.append(request)
        if len(calls) < 3:
            return httpx.Response(429 if len(calls) == 1 else 503, headers={"Retry-After": "2"})
        return httpx.Response(200, json={"primaryAccession": "P00533"})

    q = make_quarry(handler)
    q.http.sleep = sleeps.append
    assert q.query("uniprot", "entry", accession="P00533").returned == 1
    assert sleeps == [2, 2]
    assert retry_after_seconds("Wed, 21 Oct 2015 07:28:00 GMT") == 0
    assert retry_after_seconds("invalid") is None


@pytest.mark.parametrize(
    ("status", "headers", "code", "retryable"),
    [
        (429, {"Retry-After": "3600"}, "rate_limited", True),
        (401, {}, "authentication_required", False),
        (403, {}, "access_denied", False),
        (404, {}, "not_found", False),
    ],
)
def test_nonretryable_and_long_waits_are_returned_promptly(
    make_quarry, status, headers, code, retryable
):
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(status, headers=headers)

    q = make_quarry(handler)
    with pytest.raises(MolQuarryError) as error:
        q.query("uniprot", "entry", accession="P00533")
    assert error.value.code == code and error.value.retryable is retryable
    assert len(calls) == 1


def test_network_failure_bounded_and_secret_free(make_quarry):
    calls = []

    def handler(request):
        calls.append(request)
        raise httpx.ReadTimeout("internal context not exposed", request=request)

    q = make_quarry(handler, max_attempts=2)
    with pytest.raises(MolQuarryError) as error:
        q.query("uniprot", "entry", accession="P00533")
    assert len(calls) == 2
    assert error.value.code == "network_error"
    assert "internal context" not in str(error.value)


@pytest.mark.parametrize(
    "response",
    [
        lambda: httpx.Response(200, text="<html>proxy login</html>"),
        lambda: httpx.Response(200, json={"errors": [{"message": "schema changed"}], "data": None}),
    ],
)
def test_invalid_and_graphql_errors_are_not_cached(make_quarry, response):
    calls = []

    def handler(request):
        calls.append(request)
        return response()

    q = make_quarry(handler, cache=True)
    for _ in range(2):
        with pytest.raises(MolQuarryError):
            q.query("opentargets", "target", ensembl_id="ENSG00000146648")
    assert len(calls) == 2
    assert not (q.home / "cache").exists()


def test_cache_retains_timestamp_and_expires(make_quarry):
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(
            200, json={"primaryAccession": "P00533"}, headers={"X-UniProt-Release": "2026_03"}
        )

    q = make_quarry(handler, cache=True)
    first = q.query("uniprot", "entry", accession="P00533")
    second = q.query("uniprot", "entry", accession="P00533")
    assert len(calls) == 1
    assert not first.provenance[0].cached and second.provenance[0].cached
    assert second.provenance[0].retrieved_at == first.provenance[0].retrieved_at
    cache_file = next((q.home / "cache").glob("*.json"))
    entry = json.loads(cache_file.read_text())
    entry["stored_at"] = 0
    cache_file.write_text(json.dumps(entry))
    assert not q.query("uniprot", "entry", accession="P00533").provenance[0].cached
    assert len(calls) == 2


def test_post_cache_distinguishes_variables(make_quarry):
    calls = []
    q = make_quarry(lambda r: calls.append(r) or httpx.Response(200, json={"data": {}}), cache=True)
    for id_ in ("id1", "id2", "id1"):
        q.http.json(
            SOURCES["opentargets"],
            "POST",
            "https://api.platform.opentargets.org/api/v4/graphql",
            body={"query": "query", "variables": {"id": id_}},
        )
    assert len(calls) == 2


def test_private_requests_retain_distinct_scientific_query_fingerprints(make_quarry):
    q = make_quarry(lambda _: httpx.Response(200, json={"items": []}), cache=True)
    responses = [
        q.http.json(
            SOURCES["chemspace"],
            "POST",
            "https://api.chem-space.com/v5/search/exact",
            body={"search": {"smiles": smiles}},
            headers={"X-API-Key": "private-key"},
            private=True,
        )
        for smiles in ["CCO", "CCC"]
    ]
    assert responses[0].provenance.request_sha256 != responses[1].provenance.request_sha256
    assert all(not r.provenance.cached for r in responses)
    assert not list((q.home / "cache").glob("*.json"))


def test_decompressed_json_is_bounded_and_not_decoded_twice(make_quarry):
    import gzip

    body = gzip.compress(b'{"primaryAccession":"P00533"}')
    q = make_quarry(
        lambda _: httpx.Response(
            200, stream=httpx.ByteStream(body), headers={"Content-Encoding": "gzip"}
        )
    )
    assert q.query("uniprot", "entry", accession="P00533").returned == 1
    limited = make_quarry(
        lambda _: httpx.Response(
            200, stream=httpx.ByteStream(body), headers={"Content-Encoding": "gzip"}
        ),
        max_response_bytes=10,
    )
    with pytest.raises(MolQuarryError) as error:
        limited.query("uniprot", "entry", accession="P00533")
    assert error.value.code == "response_too_large"


def test_text_and_json_caches_do_not_collide(make_quarry):
    calls = []
    q = make_quarry(lambda r: calls.append(r) or httpx.Response(200, json={"x": 1}), cache=True)
    args = (SOURCES["chebi"], "GET", "https://www.ebi.ac.uk/fixture")
    assert isinstance(q.http.json(*args).data, dict)
    assert isinstance(q.http.text(*args).data, str)
    assert len(calls) == 2


def test_bulk_redirect_rejects_external_host(make_quarry):
    calls = []
    q = make_quarry(
        lambda r: (
            calls.append(r)
            or httpx.Response(302, headers={"location": "https://127.0.0.1/private"})
        )
    )
    with pytest.raises(MolQuarryError) as error:
        q.query("biolip", "files")
    assert error.value.code == "invalid_redirect" and len(calls) == 1
