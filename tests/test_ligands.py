import pytest

from molquarry.workflows.inhibitors import literature_queries
from molquarry.workflows.ligands import ligand_inventory, measurement_class


@pytest.mark.parametrize(
    ("endpoint", "relation", "value", "unit", "text", "expected"),
    [
        ("Kd", "=", "10", "nM", "", "quantitative_potency_reported"),
        ("IC50", ">", "10000", "nM", "", "lower_bound_only"),
        ("Kd", None, None, None, "Not Active", "reported_inactive"),
        ("Inhibition", "=", "99", "%", "", "other_assay_readout"),
        ("IC50", None, "10", "nM", "", "relation_unresolved"),
        ("IC50", "=", "nan", "nM", "", "other_readout_or_invalid_value"),
    ],
)
def test_evidence_buckets_preserve_negatives_and_endpoints(
    endpoint, relation, value, unit, text, expected
):
    assert measurement_class(endpoint, relation, value, unit, text) == expected


def test_ligand_mode_does_not_require_inhibitor_vocabulary():
    query = literature_queries(
        [{"input": "STK17B", "gene": "STK17B", "aliases": ["DRAK2"]}], "2026-09-19", "ligand"
    )["STK17B"]
    assert 'TITLE_ABS:"DRAK2"' in query and "inhibit*" not in query


def test_bindingdb_censoring_and_duplicate_source_records_are_retained():
    pytest.importorskip("rdkit")
    raw = {
        "monomerid": "1",
        "smile": "CCO",
        "affinity_type": "Kd",
        "affinity": ">1.0e+4",
        "pmid": "1234",
        "doi": "10.fixture",
    }
    dossier = {
        "chembl_details": {"molecule": [], "document": []},
        "activities": [],
        "bindingdb": [{"input_target": "TEST", "record": raw}] * 2,
    }
    result = ligand_inventory(dossier)
    assert len(result["compounds"]) == 1 and len(result["measurements"]) == 2
    assert result["measurements"][0]["relation"] == ">"
    assert not result["compounds"][0]["has_reported_quantitative_potency"]
    assert result["compounds"][0]["measurement_count"] == 2
