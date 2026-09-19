import json
from urllib.parse import parse_qs

import httpx
import pytest

from molquarry import MolQuarryError


@pytest.mark.parametrize(
    ("source", "operation", "params", "path", "body"),
    [
        (
            "bindingdb",
            "by_uniprot",
            {"uniprot": "P00533"},
            "/rest/getLigandsByUniprots",
            {
                "getLindsByUniprotsResponse": {
                    "affinities": [{"affinity_type": "Ki", "affinity": "<10"}]
                }
            },
        ),
        (
            "chebi",
            "search",
            {"query": "aspirin"},
            "/chebi/backend/api/public/es_search/",
            {
                "results": [{"_source": {"chebi_accession": "CHEBI:15365"}}],
                "total": 1,
                "number_pages": 1,
            },
        ),
        (
            "chebi",
            "compound",
            {"chebi_id": "CHEBI:15365"},
            "/chebi/backend/api/public/compound/15365/",
            {"id": 15365},
        ),
        (
            "unichem",
            "mapping",
            {"compound": "CHEMBL25", "source_id": 1},
            "/unichem/api/v1/compounds",
            {"compounds": [{"uci": 1}]},
        ),
        (
            "gpcrdb",
            "protein",
            {"entry_name": "adrb2_human"},
            "/services/protein/adrb2_human/",
            {"sequence": "ACDE"},
        ),
        (
            "klifs",
            "kinases",
            {"name": "EGFR"},
            "/api/kinase_names",
            [{"kinase_ID": 406, "name": "EGFR", "species": "Human"}],
        ),
        ("hpa", "search", {"query": "EGFR"}, "/api/search_download.php", [{"Gene": "EGFR"}]),
        (
            "gtex",
            "genes",
            {"query": "EGFR"},
            "/api/v2/reference/geneSearch",
            {
                "data": [{"gencodeId": "ENSG00000146648.20"}],
                "paging_info": {"totalNumberOfItems": 1},
            },
        ),
        (
            "string",
            "network",
            {"identifiers": ["EGFR", "GRB2"]},
            "/api/json/network",
            [{"score": 0.8}],
        ),
        (
            "reactome",
            "pathways_by_uniprot",
            {"identifier": "P00533"},
            "/ContentService/data/mapping/UniProt/P00533/pathways",
            [{"stId": "R-HSA-177929"}],
        ),
        (
            "clinpgx",
            "chemical",
            {"name": "warfarin"},
            "/v1/data/chemical",
            {"data": [{"id": "PA451906"}], "status": "success"},
        ),
        (
            "coconut",
            "search",
            {"query": "caffeine"},
            "/api/search",
            {"data": {"data": [{"name": "caffeine"}], "total": 1}},
        ),
        (
            "lotus",
            "search",
            {"query": "LTS0253154"},
            "/api/search/simple",
            {"naturalProducts": [{"lotus_id": "LTS0253154"}]},
        ),
        (
            "drugsfda",
            "search",
            {"query": "aspirin"},
            "/drug/drugsfda.json",
            {
                "meta": {"last_updated": "2026-09-18", "results": {"total": 1}},
                "results": [{"application_number": "NDA000001"}],
            },
        ),
        (
            "orangebook",
            "search",
            {"query": "aspirin"},
            "/drug/orangebook.json",
            {"meta": {"results": {"total": 1}}, "results": [{"product_number": "001"}]},
        ),
        (
            "surechembl",
            "chemical",
            {"identifier": "SCHEMBL1353"},
            "/api/chemical/id/1353",
            {"status": "OK", "data": [{"chemical_id": "1353"}]},
        ),
    ],
)
def test_public_response_contracts(make_quarry, source, operation, params, path, body):
    def handle(request):
        assert request.url.path == path
        if source == "unichem":
            assert json.loads(request.content)["sourceID"] == 1
        if source == "string":
            form = parse_qs(request.content.decode())
            assert form["add_nodes"] == ["0"]
            assert form["identifiers"] == ["EGFR\rGRB2"]
        return httpx.Response(200, json=body)

    result = make_quarry(handle).query(source, operation, **params)
    assert result.returned == 1
    assert result.provenance[0].url.startswith("https://")
    if source == "bindingdb":
        assert result.records[0]["affinity"] == "<10"


def test_gtex_dataset_pagination_and_empty_are_distinct(make_quarry):
    pages = []

    def handle(request):
        page = int(request.url.params["page"])
        pages.append(page)
        assert request.url.params["datasetId"] == "gtex_v10"
        return httpx.Response(
            200,
            json={
                "data": [{"tissueSiteDetailId": f"t{page}"}],
                "paging_info": {"totalNumberOfItems": 2},
            },
        )

    q = make_quarry(handle)
    results = list(q.iter_pages("gtex", "expression", gencode_id="ENSG00000146648.20", limit=1))
    assert pages == [0, 1] and results[-1].next_parameters is None


AUTH_CASES = [
    ("gtopdb", "ligands", {"query": "aspirin"}, "GTOPDB_API_KEY", "GTP-API-Key", [{"ligandId": 1}]),
    (
        "enamine",
        "exact",
        {"smiles_list": ["CCO"]},
        "REAL_API_KEY",
        "X-API-KEY",
        [{"smiles": "CCO"}],
    ),
    (
        "chemspace",
        "search",
        {"smiles": "CCO"},
        "CHEMSPACE_API_KEY",
        "X-API-Key",
        {"items": [{"csId": "CS1"}], "filtered": 1},
    ),
    (
        "molport",
        "availability",
        {"identifiers": ["CCO"]},
        "MOLPORT_API_KEY",
        "X-API-Key",
        {"search_key": "ABC"},
    ),
    (
        "lens",
        "search",
        {"query": "EGFR"},
        "LENS_API_TOKEN",
        "Authorization",
        {"data": [{"lens_id": "1"}], "total": 1},
    ),
    (
        "comptox",
        "chemical",
        {"query": "toluene"},
        "COMPTOX_API_KEY",
        "x-api-key",
        [{"dtxsid": "DTXSID7021360"}],
    ),
    (
        "uspto",
        "application",
        {"application_number": "16123456"},
        "USPTO_API_KEY",
        "X-API-KEY",
        {"count": 1},
    ),
]


@pytest.mark.parametrize(("source", "op", "params", "env", "header", "body"), AUTH_CASES)
def test_auth_fails_before_network_then_uses_uncached_header(
    make_quarry, monkeypatch, tmp_path, source, op, params, env, header, body
):
    monkeypatch.delenv(env, raising=False)
    requests = []

    def handle(request):
        requests.append(request)
        assert "secret-value" in request.headers[header]
        assert "secret-value" not in str(request.url)
        if source == "chemspace":
            assert request.url.path == "/v5/search/exact"
            assert json.loads(request.content)["search"]["shipToCountry"] == "US"
        if source == "molport":
            assert "orders" not in request.url.path
        return httpx.Response(200, json=body, headers={"Set-Cookie": "sensitive-cookie"})

    q = make_quarry(handle, cache=True)
    with pytest.raises(MolQuarryError) as error:
        q.query(source, op, **params)
    assert error.value.code == "authentication_required" and not requests
    monkeypatch.setenv(env, "secret-value")
    for _ in range(2):
        result = q.query(source, op, **params)
        assert result.returned == 1
        assert "secret-value" not in result.model_dump_json()
    assert len(requests) == 2
    assert not list((tmp_path / "cache").glob("*.json"))


def test_epo_oauth_and_namespace_preservation(make_quarry, monkeypatch):
    monkeypatch.setenv("EPO_CONSUMER_KEY", "client-id")
    monkeypatch.setenv("EPO_CONSUMER_SECRET", "private-secret")

    def handle(request):
        if request.url.path.endswith("accesstoken"):
            assert request.headers["authorization"].startswith("Basic ")
            return httpx.Response(200, json={"access_token": "private-token"})
        assert request.headers["authorization"] == "Bearer private-token"
        assert request.url.params["Range"] == "1-1"
        return httpx.Response(
            200,
            text='<ops:world xmlns:ops="http://ops.epo.org">'
            '<ops:biblio-search total-result-count="2"><ops:publication-reference>'
            "<ops:document-id>EP1000000</ops:document-id></ops:publication-reference>"
            "</ops:biblio-search></ops:world>",
        )

    r = make_quarry(handle, cache=True).query("epo", "search", query="ti=kinase", limit=1)
    assert r.total == 2 and r.next_parameters["offset"] == 1
    assert "EP1000000" in r.records[0]["xml"] and "private-token" not in r.model_dump_json()


def test_bigquery_plan_is_parameterized_and_dry_run_default(make_quarry, monkeypatch):
    secret_query = "' OR 1=1; DROP TABLE x; --"
    requests = []

    def handle(request):
        requests.append(request)
        body = json.loads(request.content)
        assert body["dryRun"] is True and body["maximumBytesBilled"] == "1024"
        assert secret_query not in body["query"]
        return httpx.Response(200, json={"totalBytesProcessed": "9000", "jobComplete": True})

    q = make_quarry(handle)
    plan = q.query("google_patents", "query_plan", keyword=secret_query)
    assert not requests and plan.records[0]["dryRun"] is True
    assert plan.provenance == []
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "test-project")
    monkeypatch.setenv("GOOGLE_ACCESS_TOKEN", "secret")
    result = q.query("google_patents", "search", keyword=secret_query, maximum_bytes_billed=1024)
    assert result.records[0]["molquarry_result_kind"] == "dry_run"


def test_bigquery_retries_reuse_request_id(make_quarry, monkeypatch):
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "test-project")
    monkeypatch.setenv("GOOGLE_ACCESS_TOKEN", "private-token")
    ids = []

    def handle(request):
        ids.append(json.loads(request.content)["requestId"])
        if len(ids) == 1:
            return httpx.Response(503)
        return httpx.Response(200, json={"jobComplete": False, "jobReference": {"jobId": "job1"}})

    q = make_quarry(handle)
    r = q.query(
        "google_patents", "search", keyword="kinase", dry_run=False, maximum_bytes_billed=1024
    )
    assert len(ids) == 2 and ids[0] == ids[1] and len(ids[0]) == 36
    assert r.records[0]["molquarry_result_kind"] == "pending_job"


def test_surechembl_family_normalization(make_quarry):
    seen = []

    def handle(request):
        seen.append(request.url.path)
        data = (
            {"US12345A1": ["US-12345-A1"]}
            if "normalized" in request.url.path
            else [{"doc_id": "US-12345-A1"}]
        )
        return httpx.Response(200, json={"status": "OK", "data": data})

    q = make_quarry(handle)
    page = q.query("surechembl", "family", publication="US12345A1")
    assert seen[-1] == "/api/document/US-12345-A1/family/members"
    assert len(page.provenance) == 2


def test_surechembl_smiles_envelope_exposes_chemical_fields(make_quarry):
    q = make_quarry(
        lambda _: httpx.Response(
            200,
            json={
                "status": "OK",
                "data": {"CCO": {"chemical_id": "1", "inchi_key": "ethanol-key"}},
            },
        )
    )
    result = q.query("surechembl", "chemical_by_smiles", smiles="CCO")
    assert result.records == [{"chemical_id": "1", "inchi_key": "ethanol-key"}]
    assert result.parameters == {"smiles": "CCO"}
