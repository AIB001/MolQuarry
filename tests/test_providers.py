import json
from urllib.parse import parse_qs

import httpx
import pytest

from molquarry import MolQuarryError


def test_pubchem_current_smiles_and_post_encoding(make_quarry):
    smiles = r"C/C=C\C.[NH4+]"

    def handler(request):
        assert request.method == "POST"
        assert "ConnectivitySMILES" in request.url.path
        assert parse_qs(request.content.decode())["smiles"] == [smiles]
        return httpx.Response(
            200,
            json={
                "PropertyTable": {
                    "Properties": [
                        {"CID": 1, "SMILES": smiles, "ConnectivitySMILES": "CC=CC.[NH4+]"}
                    ]
                }
            },
        )

    page = make_quarry(handler).query(
        "pubchem", "properties", identifier=smiles, namespace="smiles"
    )
    assert page.records[0]["SMILES"] == smiles
    assert page.provenance[0].method == "POST"


def test_pubchem_batch_missing_keys_and_vendor_categories(make_quarry):
    key = "BSYNRYMUTXBXSQ-UHFFFAOYSA-N"

    def handler(request):
        if "/categories/" in request.url.path:
            return httpx.Response(
                200,
                json={
                    "SourceCategories": {
                        "RecordType": "CID",
                        "RecordNumber": 2244,
                        "Categories": [{"Category": "Chemical Vendors", "Sources": []}],
                    }
                },
            )
        assert request.method == "POST"
        assert parse_qs(request.content.decode())["inchikey"] == [key]
        return httpx.Response(
            200, json={"PropertyTable": {"Properties": [{"CID": 2244, "InChIKey": key}]}}
        )

    q = make_quarry(handler)
    assert q.query("pubchem", "batch_properties", inchikeys=[key]).records[0]["CID"] == 2244
    assert (
        q.query("pubchem", "source_categories", cid=2244).records[0]["Category"]
        == "Chemical Vendors"
    )


def test_chembl_batch_details_retains_continuation_and_all_ids(make_quarry):
    def handler(request):
        assert request.url.params["molecule_chembl_id__in"] == "CHEMBL25,CHEMBL26"
        return httpx.Response(
            200,
            json={
                "molecules": [{"molecule_chembl_id": "CHEMBL26"}],
                "page_meta": {"offset": 0, "limit": 1, "total_count": 2, "next": "/molecule"},
            },
        )

    page = make_quarry(handler).query(
        "chembl", "batch_details", entity="molecule", chembl_ids=["CHEMBL25", "CHEMBL26"], limit=1
    )
    assert page.next_parameters["chembl_ids"] == ["CHEMBL25", "CHEMBL26"]
    assert page.next_parameters["offset"] == 1


def test_pubchem_similarity_and_assay(make_quarry):
    def handler(request):
        if "/assay/" in request.url.path:
            return httpx.Response(
                200, json={"PC_AssayContainer": [{"assay": {"descr": {"aid": {"id": 1}}}}]}
            )
        assert request.url.params["Threshold"] == "95"
        assert request.url.params["MaxRecords"] == "2"
        return httpx.Response(200, json={"IdentifierList": {"CID": [2244, 1]}})

    q = make_quarry(handler)
    page = q.query("pubchem", "similarity", smiles="CC", threshold=95, limit=2)
    assert page.records == [{"CID": 2244}, {"CID": 1}]
    assert page.total is None and page.warnings
    assert q.query("pubchem", "assay", aid=1).returned == 1


def test_pubchem_substructure_preserves_matching_options_and_coverage(make_quarry):
    smiles = r"C/C=C\C.[NH4+]"

    def handler(request):
        assert request.method == "POST"
        assert request.url.path.endswith("/fastsubstructure/smiles/cids/JSON")
        assert parse_qs(request.content.decode())["smiles"] == [smiles]
        assert request.url.params["Stereo"] == "relative"
        assert request.url.params["MatchCharges"] == "false"
        assert request.url.params["MatchIsotopes"] == "true"
        assert request.url.params["RingsNotEmbedded"] == "true"
        assert request.url.params["MaxRecords"] == "2"
        return httpx.Response(200, json={"IdentifierList": {"CID": [2244, 1]}})

    page = make_quarry(handler).query(
        "pubchem",
        "substructure",
        smiles=smiles,
        stereo="relative",
        match_charges=False,
        rings_not_embedded=True,
        limit=2,
    )
    assert page.records == [{"CID": 2244}, {"CID": 1}]
    assert page.total is None and page.next_parameters is None and page.warnings
    assert page.parameters["match_isotopes"] is True
    assert page.provenance[0].method == "POST"


@pytest.mark.parametrize("parameters", [{"stereo": "typo"}, {"limit": 101}, {"smarts": "C*"}])
def test_pubchem_substructure_rejects_unsupported_options(make_quarry, parameters):
    quarry = make_quarry(lambda _: pytest.fail("Invalid query must fail before network I/O"))
    with pytest.raises(MolQuarryError) as error:
        quarry.query("pubchem", "substructure", smiles="CC", **parameters)
    assert error.value.code == "invalid_parameters"


def test_chembl_pagination_preserves_assay_and_censored_relation(make_quarry):
    requests = []
    record = {
        "activity_id": 42,
        "standard_type": "IC50",
        "standard_relation": "<",
        "standard_value": "10",
        "standard_units": "nM",
        "assay_chembl_id": "CHEMBL12",
        "assay_type": "B",
        "document_chembl_id": "CHEMBL13",
    }

    def handler(request):
        requests.append(request)
        offset = int(request.url.params["offset"])
        assert request.url.params["target_chembl_id"] == "CHEMBL203"
        assert request.url.params["standard_relation"] == "<"
        return httpx.Response(
            200,
            json={
                "activities": [record],
                "page_meta": {
                    "offset": offset,
                    "limit": 1,
                    "total_count": 2,
                    "next": "/next" if offset == 0 else None,
                },
            },
        )

    q = make_quarry(handler)
    pages = list(
        q.iter_pages(
            "chembl",
            "activities",
            target_chembl_id="CHEMBL203",
            standard_type="IC50",
            standard_relation="<",
            limit=1,
        )
    )
    assert len(pages) == 2
    assert pages[0].records == [
        record
    ]  # No recoding '<10' to an equality or converting IC50 to Kd.
    assert pages[1].next_parameters is None
    assert [r.url.params["offset"] for r in requests] == ["0", "1"]


@pytest.mark.parametrize(
    ("op", "params", "path", "payload"),
    [
        (
            "molecule",
            {"chembl_id": "CHEMBL25"},
            "/molecule/CHEMBL25.json",
            {"molecule_chembl_id": "CHEMBL25"},
        ),
        (
            "target",
            {"chembl_id": "CHEMBL203"},
            "/target/CHEMBL203.json",
            {"target_chembl_id": "CHEMBL203"},
        ),
        (
            "search_molecules",
            {"query": "aspirin"},
            "/molecule/search.json",
            {"molecules": [], "page_meta": {"total_count": 0, "next": None}},
        ),
        (
            "targets_by_uniprot",
            {"uniprot": "P00533"},
            "/target.json",
            {
                "targets": [{"target_chembl_id": "CHEMBL203"}],
                "page_meta": {"total_count": 1, "next": None},
            },
        ),
    ],
)
def test_chembl_mapping_and_search(make_quarry, op, params, path, payload):
    def handler(request):
        assert request.url.path.endswith(path)
        if op == "targets_by_uniprot":
            assert request.url.params["target_components__accession"] == "P00533"
        if op == "search_molecules":
            assert request.url.params["q"] == "aspirin"
        return httpx.Response(200, json=payload)

    make_quarry(handler).query("chembl", op, **params)


def test_uniprot_next_link_release_and_total(make_quarry):
    seen = []

    def handler(request):
        seen.append(request)
        if len(seen) == 1:
            return httpx.Response(
                200,
                json={"results": [{"primaryAccession": "P00533"}]},
                headers={
                    "X-UniProt-Release": "2026_03",
                    "X-Total-Results": "2",
                    "Link": (
                        "<https://rest.uniprot.org/uniprotkb/search?cursor=opaque%2Btoken>"
                        '; rel="next"'
                    ),
                },
            )
        assert request.url.params["cursor"] == "opaque+token"
        assert request.url.params["query"] == "reviewed:true"
        return httpx.Response(200, json={"results": [{"primaryAccession": "P04637"}]})

    q = make_quarry(handler)
    pages = list(q.iter_pages("uniprot", "search", query="reviewed:true", limit=1))
    assert len(pages) == 2
    assert pages[0].provenance[0].source_version == "2026_03"
    assert pages[0].total == 2


def test_rcsb_204_no_hits_and_uniprot_query(make_quarry):
    def handler(request):
        query = json.loads(request.content)
        assert query["request_options"]["results_content_type"] == ["experimental"]
        assert query["query"]["nodes"][0]["parameters"]["value"] == "P00533"
        return httpx.Response(204)

    page = make_quarry(handler).query("rcsb", "by_uniprot", uniprot="P00533")
    assert page.records == [] and page.total == 0


@pytest.mark.parametrize(
    ("source", "op", "params", "path", "record"),
    [
        ("rcsb", "entry", {"pdb_id": "7knw"}, "/entry/7KNW", {"rcsb_id": "7KNW"}),
        ("rcsb", "ligand", {"ccd_id": "ATP"}, "/chemcomp/ATP", {"chem_comp": {"id": "ATP"}}),
        (
            "uniprot",
            "entry",
            {"accession": "P00533"},
            "/P00533.json",
            {"primaryAccession": "P00533"},
        ),
        (
            "clinicaltrials",
            "study",
            {"nct_id": "NCT03257124"},
            "/NCT03257124",
            {"protocolSection": {"identificationModule": {"nctId": "NCT03257124"}}},
        ),
        (
            "mcule",
            "compound",
            {"mcule_id": "MCULE-9380369173"},
            "/compound/MCULE-9380369173/",
            {"mcule_id": "MCULE-9380369173"},
        ),
    ],
)
def test_record_lookup_contracts(make_quarry, source, op, params, path, record):
    def handler(request):
        assert request.url.path.endswith(path)
        return httpx.Response(200, json=record)

    assert make_quarry(handler).query(source, op, **params).records == [record]


def test_rcsb_search_continuation(make_quarry):
    def handler(request):
        body = json.loads(request.content)
        assert body["query"]["service"] == "full_text"
        return httpx.Response(
            200, json={"total_count": 2, "result_set": [{"identifier": "7KNW", "score": 1.0}]}
        )

    page = make_quarry(handler).query("rcsb", "search", query="EGFR", limit=1)
    assert page.next_parameters == {"query": "EGFR", "offset": 1, "limit": 1}


def test_clinicaltrials_cursor_keeps_filters(make_quarry):
    def handler(request):
        assert request.url.params["query.term"] == "EGFR"
        assert request.url.params["filter.overallStatus"] == "RECRUITING"
        assert request.url.params["countTotal"] == "true"
        return httpx.Response(
            200,
            json={
                "totalCount": 99,
                "studies": [{"protocolSection": {}}],
                "nextPageToken": "opaque",
            },
        )

    q = make_quarry(handler)
    page = q.query("clinicaltrials", "search", query="EGFR", status="RECRUITING", limit=1)
    assert page.next_parameters["status"] == "RECRUITING"
    assert page.next_parameters["page_token"] == "opaque"
    assert page.total == 99


@pytest.mark.parametrize("op", ["target", "search", "diseases"])
def test_opentargets_queries_use_variables(make_quarry, op):
    def handler(request):
        body = json.loads(request.content)
        assert request.method == "POST"
        assert "variables" in body
        if op == "target":
            data = {"target": {"id": body["variables"]["id"], "approvedSymbol": "EGFR"}}
        elif op == "search":
            assert body["variables"]["q"] == 'EGFR "quoted"'
            data = {"search": {"total": 2, "hits": [{"id": "ENSG00000146648"}]}}
        else:
            data = {
                "target": {
                    "associatedDiseases": {
                        "count": 2,
                        "rows": [{"score": 0.5, "disease": {"id": "EFO_1"}}],
                    }
                }
            }
        return httpx.Response(200, json={"data": data})

    params = (
        {"query": 'EGFR "quoted"', "limit": 1}
        if op == "search"
        else {"ensembl_id": "ENSG00000146648"}
    )
    if op == "diseases":
        params["limit"] = 1
    page = make_quarry(handler).query("opentargets", op, **params)
    assert page.returned == 1
    if op != "target":
        assert page.next_parameters["page"] == 1


def test_opentargets_null_entity_is_not_success(make_quarry):
    q = make_quarry(lambda _: httpx.Response(200, json={"data": {"target": None}}))
    with pytest.raises(MolQuarryError, match="not found"):
        q.query("opentargets", "target", ensembl_id="ENSG00000146648")


def test_alphafold_resolves_current_version_and_multiple_models(make_quarry):
    models = [
        {
            "modelEntityId": "AF-P00533-F1",
            "latestVersion": 6,
            "cifUrl": "https://alphafold.ebi.ac.uk/files/AF-P00533-F1-model_v6.cif",
        }
    ]
    q = make_quarry(lambda _: httpx.Response(200, json=models))
    assert q.query("alphafold", "prediction", accession="P00533").records == models
    plan = q.plan_download("alphafold", "structure", accession="P00533")
    assert plan.url.endswith("_v6.cif") and plan.source_version == "6"
    models.append({**models[0], "modelEntityId": "AF-P00533-F2"})
    with pytest.raises(MolQuarryError) as error:
        q.plan_download("alphafold", "structure", accession="P00533")
    assert error.value.code == "ambiguous_model"
    assert q.plan_download("alphafold", "structure", accession="P00533", model_id="AF-P00533-F1")


@pytest.mark.parametrize("enveloped", [False, True])
def test_mcule_live_and_documented_catalog_shapes(make_quarry, enveloped):
    catalog = {
        "id": 1,
        "last_updated": "2026-08-01",
        "files": [
            {
                "filename": "data.smi.gz",
                "file_type": "smi.gz",
                "size_mb": 2408,
                "download_url": "https://dl.mcule.com/database/data.smi.gz",
                "sha256_checksum": "a" * 64,
            }
        ],
    }
    payload = {"results": [catalog], "count": 1, "next": None} if enveloped else catalog
    q = make_quarry(lambda _: httpx.Response(200, json=payload))
    assert q.query("mcule", "database_files", dataset_id=1).records == [catalog]
    plan = q.plan_download("mcule", "dataset", dataset_id=1)
    assert plan.source_version == "2026-08-01" and plan.expected_sha256 == "a" * 64
    assert plan.estimated_bytes == 2_408_000_000


def test_mcule_lookup(make_quarry):
    q = make_quarry(lambda _: httpx.Response(200, json={"results": [{"mcule_id": "MCULE-1"}]}))
    assert q.query("mcule", "lookup", inchikey="BSYNRYMUTXBXSQ-UHFFFAOYSA-N").returned == 1


def test_alphafold_selects_exact_accession_without_merging_isoforms(make_quarry):
    models = [
        {
            "modelEntityId": "AF-P00533-2-F1",
            "uniprotAccession": "P00533-2",
            "latestVersion": 6,
            "cifUrl": "https://alphafold.ebi.ac.uk/files/isoform.cif",
        },
        {
            "modelEntityId": "AF-P00533-F1",
            "uniprotAccession": "P00533",
            "latestVersion": 6,
            "cifUrl": "https://alphafold.ebi.ac.uk/files/canonical.cif",
        },
    ]
    q = make_quarry(lambda _: httpx.Response(200, json=models))
    assert q.plan_download("alphafold", "structure", accession="P00533").filename == "canonical.cif"
    assert q.plan_download("alphafold", "structure", accession="P00533-2").filename == "isoform.cif"
    assert q.query("alphafold", "prediction", accession="P00533").returned == 2


def test_clinicaltrials_continuation_can_omit_total_count(make_quarry):
    def handler(request):
        if "pageToken" in request.url.params:
            return httpx.Response(200, json={"studies": [{"protocolSection": {}}]})
        return httpx.Response(
            200,
            json={"studies": [{"protocolSection": {}}], "totalCount": 2, "nextPageToken": "next"},
        )

    pages = list(make_quarry(handler).iter_pages("clinicaltrials", "search", query="EGFR", limit=1))
    assert len(pages) == 2
    assert pages[0].total == 2 and pages[1].total is None
