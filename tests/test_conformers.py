import json

import pytest

from molquarry.workflows.compounds import properties
from molquarry.workflows.conformers import export_inventory_conformers, saved_result_valid

pytest.importorskip("rdkit")


def test_resume_repairs_deleted_and_corrupted_sdf_files(tmp_path):
    _, prop = properties("C[C@H](O)C(=O)O")
    row = {"compound_id": prop["inchikey"], **prop, "chemical_status": "parsed", "pmids": []}
    source = tmp_path / "inventory.json"
    source.write_text(json.dumps({"compounds": [row]}))
    output = tmp_path / "conformers"
    result = export_inventory_conformers(source, output, workers=1)
    assert result["written"] == 1 and saved_result_valid(output, row)
    path = output / result["records"][0]["file"]
    path.unlink()
    assert not saved_result_valid(output, row)
    export_inventory_conformers(source, output, workers=1)
    assert saved_result_valid(output, row)
    path.write_text("corrupt data")
    assert not saved_result_valid(output, row)
    export_inventory_conformers(source, output, workers=1)
    assert saved_result_valid(output, row)


def test_inventory_duplicate_ids_rejected_before_output(tmp_path):
    row = {"compound_id": "duplicate", "chemical_status": "parsed"}
    source = tmp_path / "inventory.json"
    source.write_text(json.dumps({"compounds": [row, row]}))
    with pytest.raises(ValueError, match="unique IDs"):
        export_inventory_conformers(source, tmp_path / "out")
    assert not (tmp_path / "out").exists()
