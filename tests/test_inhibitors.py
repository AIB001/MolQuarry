"""Contracts that prevent false inhibitor claims and loss of incomplete-source evidence."""

import hashlib
from types import SimpleNamespace

import httpx
import pytest
from pydantic import ValidationError

from molquarry import MolQuarryError
from molquarry.models import QueryResult
from molquarry.workflows.inhibitors import (
    Collector,
    InhibitorSearch,
    collect_inhibitor_evidence,
    literature_queries,
)
from molquarry.workflows.review import (
    Candidate,
    Curation,
    Measurement,
    integrate_review,
    normalized_measurement,
)


def test_literature_cursor_keeps_query_and_empty_continuation_proves_exhaustion(make_quarry):
    seen = []

    def handler(request):
        params = request.url.params
        seen.append(params["cursorMark"])
        assert params["query"] == "TITLE_ABS:SND1"
        assert params["resultType"] == "core"
        if len(seen) == 1:
            return httpx.Response(
                200,
                json={
                    "hitCount": 1,
                    "nextCursorMark": "opaque+/=",
                    "resultList": {"result": [{"id": "35121987", "source": "MED"}]},
                },
            )
        return httpx.Response(
            200, json={"hitCount": 1, "nextCursorMark": "opaque+/=", "resultList": {"result": []}}
        )

    pages = list(make_quarry(handler).iter_pages("europepmc", "search", query="TITLE_ABS:SND1"))
    assert seen == ["*", "opaque+/="]
    assert pages[-1].next_parameters is None
    assert sum(p.returned for p in pages) == 1


def test_fulltext_accept_header_tables_locators_and_untrusted_links(make_quarry):
    def handler(request):
        assert request.headers["accept"] == "application/xml"
        assert request.url.path.endswith("/PMC8818087/fullTextXML")
        return httpx.Response(
            200,
            text="""<article xmlns:xlink="http://www.w3.org/1999/xlink">
        <body><sec><title>Results</title><p id="p1">C26 binds SND1.</p>
        <table-wrap id="t1"><table><tr><th>Compound</th><th>IC50</th></tr>
        <tr><td>Suramin</td><td>0.6</td></tr></table></table-wrap>
        <supplementary-material id="s1"><media xlink:href="https://untrusted.test/execute"/>
        </supplementary-material></sec></body></article>""",
        )

    q = make_quarry(handler)
    page = q.query("europepmc", "fulltext", pmcid="PMC8818087", offset=1, limit=1)
    assert page.total == 3 and page.next_parameters["offset"] == 2
    row = page.records[0]
    assert row["block_id"] == "t1" and row["section"] == ["Results"]
    assert row["tables"][0][1][1]["text"] == "0.6"
    assert row["locator"] == "/article/body[1]/sec[1]/table-wrap[1]"
    assert q.query("europepmc", "fulltext", pmcid="PMC8818087", contains="C26").returned == 1


@pytest.mark.parametrize("payload", ["<html>Login</html>", "<broken"])
def test_bad_fulltext_is_not_zero_hits(make_quarry, payload):
    q = make_quarry(lambda _: httpx.Response(200, text=payload))
    with pytest.raises(MolQuarryError, match="schema") as exc:
        q.query("europepmc", "fulltext", pmcid="PMC1")
    assert exc.value.code == "invalid_response"


def test_assay_mentions_and_assay_document_activity_filters(make_quarry):
    def handler(request):
        params = request.url.params
        if "/assay.json" in request.url.path:
            assert params["description__icontains"] == "MTDH"
            key = "assays"
            row = {"assay_chembl_id": "CHEMBL1", "target_chembl_id": "CHEMBL612545"}
        else:
            assert (
                params.get("assay_chembl_id") == "CHEMBL1"
                or params.get("document_chembl_id") == "CHEMBL2"
            )
            key, row = (
                "activities",
                {"standard_type": "IC50", "standard_relation": "<", "standard_value": "10"},
            )
        return httpx.Response(200, json={key: [row], "page_meta": {"total_count": 1, "next": None}})

    q = make_quarry(handler)
    assert (
        q.query("chembl", "search_assays", query="MTDH").records[0]["target_chembl_id"]
        == "CHEMBL612545"
    )
    for key, val in [("assay_chembl_id", "CHEMBL1"), ("document_chembl_id", "CHEMBL2")]:
        assert q.query("chembl", "activities", **{key: val}).records[0]["standard_relation"] == "<"


def page(records, total=None, next_parameters=None):
    return QueryResult(
        source="uniprot",
        operation="search",
        parameters={},
        records=records,
        returned=len(records),
        total=total,
        next_parameters=next_parameters,
        provenance=[],
    )


def test_collector_preserves_partial_data_and_failure(tmp_path):
    def pages(*args, **kwargs):
        yield page([{"id": "1"}], total=2, next_parameters={"cursor": "2"})
        raise MolQuarryError("rate_limited", "Retry later", retryable=True)

    c = Collector(SimpleNamespace(iter_pages=pages), tmp_path, 10)
    assert c.query("x", "uniprot", "search") == [{"id": "1"}]
    assert c.coverage[0]["status"] == "partial_error"
    assert c.coverage[0]["next_parameters"] == {"cursor": "2"}
    raw = tmp_path / c.coverage[0]["files"][0]
    assert hashlib.sha256(raw.read_bytes()).hexdigest() == c.artifacts[0]["sha256"]


def test_collector_page_bound_remains_truncated(tmp_path):
    q = SimpleNamespace(iter_pages=lambda *a, **kw: iter([page([{"id": "1"}], 8, {"offset": 1})]))
    c = Collector(q, tmp_path, 1)
    c.query("x", "uniprot", "search")
    assert c.coverage[0]["status"] == "truncated" and c.coverage[0]["total"] == 8


def test_ambiguous_protein_stops_downstream_claims_and_refuses_overwrite(tmp_path):
    calls = []

    def pages(source, op, **kw):
        calls.append((source, op))
        yield page(
            [
                {"primaryAccession": accession, "organism": {"taxonId": 9606}}
                for accession in ["P00533", "P04637"]
            ],
            2,
        )

    q = SimpleNamespace(iter_pages=pages)
    config = InhibitorSearch(targets=["AMBIGUOUS"])
    result = collect_inhibitor_evidence(q, config, tmp_path / "run")
    assert calls == [("uniprot", "search")]
    assert (
        result["status"] == "needs_target_resolution" and result["automatic_inhibitor_claims"] == 0
    )
    with pytest.raises(FileExistsError):
        collect_inhibitor_evidence(q, config, tmp_path / "run")


def test_alias_query_quotes_literal_and_has_all_target_interaction_scope():
    targets = [
        {"input": "MTDH", "gene": "MTDH", "aliases": ['x" OR *:*']},
        {"input": "SND1", "gene": "SND1", "aliases": ["Tudor-SN"]},
    ]
    queries = literature_queries(targets, "2026-09-19")
    assert 'TITLE_ABS:"x\\" OR *:*"' in queries["MTDH"]
    assert 'TITLE_ABS:"SND1"' in queries["interaction"]
    assert "FIRST_PDATE:[1900-01-01 TO 2026-09-19]" in queries["MTDH"]


def ref(source="europepmc", record="39792778", locator="abstractText"):
    return {
        "source": source,
        "record_id": record,
        "locator": locator,
        "url": "https://example.org/evidence",
    }


def measure(value=279.0, unit="nM", **kw):
    return Measurement(
        endpoint="Kd",
        value=value,
        unit=unit,
        assay="MST",
        context="SND1 purified",
        reference=ref(),
        comparison_group="same-experiment",
        **kw,
    )


def test_normalization_retains_endpoint_and_censoring():
    row = normalized_measurement(measure(10.0, "nM", relation="<"))
    assert row["value_molar"] == 1e-8
    assert row["relation"] == "<" and row["endpoint"] == "Kd" and row["unit"] == "nM"
    temperature = Measurement(
        endpoint="Delta Tm",
        value=-0.9,
        unit="degrees C",
        assay="DSF",
        context="selectivity panel",
        reference=ref(),
    )
    assert "value_molar" not in normalized_measurement(temperature)


def candidate(label="C19", **overrides):
    return Candidate.model_validate(
        {
            "candidate_id": "doi:one#" + label,
            "label": label,
            "modality": "small_molecule",
            "mechanism": "ppi_disruption",
            "status": "supported",
            "evidence_level": "abstract_reviewed",
            "target_accessions": ["Q7KZF4"],
            "references": [ref()],
            "measurements": [],
            **overrides,
        }
    )


def dossier():
    return {
        "targets": [{"accession": "Q7KZF4"}],
        "papers": [{"source": "MED", "id": "39792778"}],
        "fulltexts": {},
        "chembl_details": {"assay": [], "document": [], "molecule": []},
        "activities": [],
        "structures": [],
        "ccd_ligands": [],
        "bindingdb": [],
        "review_queue": [{"source": "MED", "id": "39792778"}, {"source": "MED", "id": "1"}],
        "coverage": [],
    }


def review(rows):
    return Curation(
        candidates=rows,
        reviewed_at="2026-09-19",
        limitations=["Abstract only"],
        papers_reviewed=[
            {"pmid": "39792778", "decision": "included", "reason": "Primary abstract"}
        ],
    )


def test_review_retains_thousandfold_conflict_and_unreviewed_papers():
    c = candidate(measurements=[measure().model_dump(), measure(279000.0).model_dump()])
    merged = integrate_review(dossier(), review([c]))
    assert len(merged["conflicts"]) == 1
    assert merged["conflicts"][0]["fold_difference"] == pytest.approx(1000)
    assert merged["conflicts"][0]["status"] == "unresolved"
    assert len(merged["candidates"][0]["measurements"]) == 2
    assert merged["unreviewed_paper_count"] == 1


def test_different_assays_and_censored_bounds_are_not_forced_into_conflicts():
    ms = [measure().model_dump(), measure(279000.0).model_dump()]
    ms[1]["comparison_group"] = "different-construct"
    assert not integrate_review(dossier(), review([candidate(measurements=ms)]))["conflicts"]
    ms[1]["comparison_group"] = "same-experiment"
    ms[1]["relation"] = ">"
    assert not integrate_review(dossier(), review([candidate(measurements=ms)]))["conflicts"]


def test_unfetched_references_and_computational_promotion_fail():
    with pytest.raises(ValueError, match="Unfetched evidence"):
        integrate_review(dossier(), review([candidate(references=[ref(record="123")])]))
    with pytest.raises(ValidationError, match="Computational-only"):
        candidate(mechanism="computational", status="supported")


def test_same_short_label_across_papers_is_not_merged():
    a, b = candidate(), candidate(candidate_id="doi:two#C19")
    merged = integrate_review(dossier(), review([a, b]))
    assert len(merged["candidates"]) == 2 and not merged["identity_groups"]
