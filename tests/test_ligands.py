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


def test_overlap_normalizes_number_format_without_mixing_targets():
    pytest.importorskip("rdkit")
    chemical = {
        "activity_id": 1,
        "molecule_chembl_id": "CHEMBL1",
        "canonical_smiles": "CCO",
        "document_chembl_id": "CHEMBL2",
        "standard_type": "Kd",
        "standard_relation": "=",
        "standard_value": "10",
        "standard_units": "nM",
    }
    binding = {
        "monomerid": "1",
        "smile": "CCO",
        "affinity_type": "Kd",
        "affinity": "10.0000",
        "pmid": "1234",
        "doi": "10.fixture",
    }
    dossier = {
        "chembl_details": {
            "molecule": [],
            "document": [{"document_chembl_id": "CHEMBL2", "pubmed_id": 1234}],
        },
        "activities": [{"input_target": "A", "record": chemical}],
        "bindingdb": [{"input_target": "A", "record": binding}],
    }
    result = ligand_inventory(dossier)
    assert all(r["possible_cross_source_overlap"] for r in result["measurements"])
    dossier["bindingdb"][0]["input_target"] = "B"
    assert not any(
        r["possible_cross_source_overlap"] for r in ligand_inventory(dossier)["measurements"]
    )
