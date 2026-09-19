import httpx
import pytest

from molquarry import MolQuarryError
from molquarry.catalog import CATEGORIES


def no_network(_):
    raise AssertionError("No network request should be issued")


def test_catalog_and_examples_are_consistent(make_quarry):
    q = make_quarry(no_network)
    all_sources = q.sources()
    assert len(all_sources) == 51
    assert len({s["id"] for s in all_sources}) == 51
    assert len(q.sources(implemented_only=True)) == 41
    for item in all_sources:
        assert set(item["categories"]) <= CATEGORIES.keys()
        descriptor = q.describe(item["id"])
        if item["implementation"] == "partial":
            assert set(descriptor["operations"]) == {"resources"}
            assert not descriptor["downloads"]
            assert descriptor["supports_local_catalog"]
        else:
            provider = q.providers[item["id"]]
            for kind in (provider.operations, provider.downloads):
                for definition in kind.values():
                    definition.input_model.model_validate(definition.example)
                    assert definition.describe()["input_schema"]["additionalProperties"] is False
    assert len(q.sources(category="patent", implemented_only=True)) == 6
    assert any(s["id"] == "surechembl" for s in q.sources(query="surechembl"))


@pytest.mark.parametrize(
    ("source", "op", "params", "code"),
    [
        ("missing", "search", {}, "unknown_source"),
        ("surechembl", "search", {}, "unknown_operation"),
        ("chembl", "missing", {}, "unknown_operation"),
        ("chembl", "activities", {}, "invalid_parameters"),
        ("chembl", "activities", {"target_id": "CHEMBL203"}, "invalid_parameters"),
        (
            "chembl",
            "activities",
            {"target_chembl_id": "CHEMBL203", "limit": 0},
            "invalid_parameters",
        ),
        ("uniprot", "entry", {"accession": "../secret"}, "invalid_parameters"),
        ("rcsb", "entry", {"pdb_id": "7KNW", "extra": "value"}, "invalid_parameters"),
        ("clinicaltrials", "search", {"query": "EGFR", "limit": "5"}, "invalid_parameters"),
    ],
)
def test_bad_requests_fail_before_network(make_quarry, source, op, params, code):
    q = make_quarry(no_network)
    with pytest.raises(MolQuarryError) as error:
        q.query(source, op, **params)
    assert error.value.code == code
    assert error.value.as_dict()["ok"] is False


def test_schema_drift_is_an_error_not_no_results(make_quarry):
    q = make_quarry(lambda _: httpx.Response(200, json={"renamed_field": []}))
    for source, op, params in [
        ("chembl", "activities", {"target_chembl_id": "CHEMBL203"}),
        ("rcsb", "search", {"query": "EGFR"}),
        ("clinicaltrials", "search", {"query": "EGFR"}),
    ]:
        with pytest.raises(MolQuarryError) as error:
            q.query(source, op, **params)
        assert error.value.code == "invalid_response"
