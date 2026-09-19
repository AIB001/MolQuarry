"""Scientific file contracts. Synthetic structures are fixtures, not AF3 inference evidence."""

import json
import sys
import zipfile

import pytest

from molquarry.workflows.af3 import import_af3_results, prepare_af3_jobs, run_local_af3
from molquarry.workflows.bundle import default_output, excel_workbook
from molquarry.workflows.compounds import prepare_molecule, properties
from molquarry.workflows.structures import kabsch, mapped_ca, pdb_export, sequence_residue_map

np = pytest.importorskip("numpy")
gemmi = pytest.importorskip("gemmi")
pytest.importorskip("rdkit")
pytest.importorskip("openpyxl")
pytest.importorskip("Bio")


def structure():
    s = gemmi.Structure()
    s.add_model(gemmi.Model(1))
    s[0].add_chain(gemmi.Chain("LONG"))
    for i, xyz in enumerate([(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)], 1):
        residue = gemmi.Residue()
        residue.name = "ALA"
        residue.seqid = gemmi.SeqId(i, " ")
        residue.label_seq = i
        residue.entity_id = "1"
        atom = gemmi.Atom()
        atom.name = "CA"
        atom.element = gemmi.Element("C")
        atom.pos = gemmi.Position(*xyz)
        residue.add_atom(atom)
        s[0][0].add_residue(residue)
    return s


def test_kabsch_recovers_transform_without_reflection():
    reference = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], float)
    rotation = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], float)
    mobile = reference @ rotation + np.array([20, -3, 8])
    r, t, rms = kabsch(mobile, reference)
    assert rms < 1e-10
    assert np.allclose(mobile @ r + t, reference)
    assert np.linalg.det(r) == pytest.approx(1)
    with pytest.raises(ValueError, match="Collinear"):
        kabsch([[0, 0, 0], [1, 0, 0], [2, 0, 0]], [[0, 0, 0], [1, 0, 0], [2, 0, 0]])


def test_sifts_mapping_uses_entity_positions_not_pdb_residue_numbers():
    s = structure()
    entity = {
        "rcsb_polymer_entity_container_identifiers": {"entity_id": "1"},
        "rcsb_polymer_entity_align": [
            {
                "reference_database_accession": "P00533",
                "reference_database_name": "UniProt",
                "aligned_regions": [{"entity_beg_seq_id": 2, "ref_beg_seq_id": 101, "length": 2}],
            }
        ],
    }
    assert set(mapped_ca(s[0][0], entity, "P00533")) == {101, 102}
    assert not mapped_ca(s[0][0], entity, "Q7KZF4")


def test_sequence_mapping_corrects_construct_deletion_inside_coarse_sifts_range():
    canonical = "ACDEFGHIKLMNPQRSTVWY"
    construct = "HHHHHH" + canonical[:7] + canonical[10:]
    entity = {
        "entity_poly": {"pdbx_seq_one_letter_code_can": construct},
        "rcsb_polymer_entity_align": [
            {
                "reference_database_accession": "P00533",
                "reference_database_name": "UniProt",
                "aligned_regions": [{"entity_beg_seq_id": 7, "ref_beg_seq_id": 1, "length": 17}],
            }
        ],
    }
    mapping, metadata = sequence_residue_map(entity, "P00533", canonical)
    assert mapping[7] == 1 and mapping[13] == 7
    assert mapping[14] == 11 and mapping[23] == 20
    assert not set(mapping) & set(range(1, 7))
    assert metadata["sequence_identity"] == 1


def test_pdb_export_maps_chain_names_and_rejects_lossy_fields(tmp_path):
    s = structure()
    path = tmp_path / "ok.pdb"
    mapping = pdb_export(s, path)
    assert mapping == {"LONG": "A"}
    parsed = gemmi.read_structure(str(path))
    assert parsed[0][0].name == "A" and parsed[0].count_atom_sites() == 4
    assert s[0][0].name == "LONG"
    s[0][0][0].name = "ABCDE"
    with pytest.raises(ValueError, match="Residue name"):
        pdb_export(s, tmp_path / "bad.pdb")
    assert not (tmp_path / "bad.pdb").exists()


def test_rdkit_preparation_preserves_stereo_identity_and_experimental_heavy_atoms():
    from rdkit import Chem

    m, desc = properties("C[C@H](O)C(=O)O")
    conformer, info = prepare_molecule(m)
    assert info["coordinates"] == "rdkit_etkdg_generated"
    assert conformer.GetConformer().Is3D()
    assert Chem.MolToInchiKey(Chem.RemoveHs(conformer)) == desc["inchikey"]
    before = np.array(Chem.RemoveHs(conformer).GetConformer().GetPositions())
    fixed, info = prepare_molecule(conformer, experimental=True)
    after = np.array(Chem.RemoveHs(fixed).GetConformer().GetPositions())
    assert np.allclose(before, after, atol=1e-9)
    assert info["coordinates"] == "experimental_heavy_atoms_fixed"


def test_excel_source_strings_are_not_formulas(tmp_path):
    from openpyxl import load_workbook

    output = tmp_path / "results.xlsx"
    excel_workbook(
        output,
        {"Candidates": [{"pmid": "0123", "source_text": '=HYPERLINK("evil")', "smiles": "CCO"}]},
    )
    book = load_workbook(output, data_only=False)
    assert book["Candidates"]["B2"].data_type == "s"
    assert book["Candidates"]["A2"].value == "0123"
    assert book["Candidates"].freeze_panes == "A2"
    with pytest.raises(FileExistsError):
        excel_workbook(output, {"Candidates": []})


def test_default_output_uses_working_directory(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HOME", "/some/other/home")
    out = default_output([{"input": "MTDH"}, {"input": "SND1"}], "inhibitor")
    assert out.parent == tmp_path and out.name.startswith("MolQuarry_MTDH_SND1_inhibitor_")


def test_af3_jobs_are_inputs_not_models_and_have_distinct_dialects(tmp_path):
    target = [{"gene": "TEST", "sequence": "ACDEFGHIK", "accession": "P00533"}]
    result = prepare_af3_jobs(target, tmp_path / "jobs")
    assert result["status"] == "prepared_not_submitted" and result["models_returned"] == 0
    server = json.loads((tmp_path / "jobs" / "alphafold_server_jobs.json").read_text())
    local = json.loads((tmp_path / "jobs" / "TEST_local.json").read_text())
    assert isinstance(server, list) and server[0]["sequences"][0]["proteinChain"]["count"] == 1
    assert local["dialect"] == "alphafold3" and local["sequences"][0]["protein"]["id"] == "A"


@pytest.mark.parametrize("name", ["../escape.cif", "/absolute.cif", "folder\\escape.cif"])
def test_af3_zip_traversal_rejected_before_writing(tmp_path, name):
    archive = tmp_path / "bad.zip"
    with zipfile.ZipFile(archive, "w") as f:
        f.writestr(name, "data_x")
    with pytest.raises(ValueError, match="Unsafe"):
        import_af3_results(archive, tmp_path / "out")
    assert not (tmp_path / "out").exists()


def test_af3_import_retains_confidence_and_marks_predictions(tmp_path):
    archive = tmp_path / "fixture.zip"
    s = structure()
    s.name = "fixture"
    with zipfile.ZipFile(archive, "w") as f:
        f.writestr("sample_model.cif", s.make_mmcif_document().as_string())
        f.writestr(
            "sample_summary_confidences.json",
            json.dumps({"ptm": 0.7, "iptm": 0.6, "ranking_score": 0.8}),
        )
    result = import_af3_results(archive, tmp_path / "out")
    assert result["status"] == "imported_predictions"
    assert result["models"][0]["experimental"] is False
    assert result["confidence"][0]["ptm"] == 0.7
    assert (tmp_path / "out" / result["models"][0]["pdb"]).is_file()


def test_af3_archive_with_cif_and_pdb_preserves_both_models(tmp_path):
    archive = tmp_path / "both.zip"
    s = structure()
    s.name = "fixture"
    with zipfile.ZipFile(archive, "w") as f:
        f.writestr("sample_model.cif", s.make_mmcif_document().as_string())
        s[0][0].name = "A"
        f.writestr("sample_model.pdb", s.make_pdb_string())
    result = import_af3_results(archive, tmp_path / "out")
    paths = [r["pdb"] for r in result["models"]]
    assert len(set(paths)) == 2 and all((tmp_path / "out" / p).exists() for p in paths)


def test_af3_invalid_zip_and_case_collisions_fail_before_output(tmp_path):
    bad = tmp_path / "bad.zip"
    bad.write_bytes(b"not a zip")
    with pytest.raises(ValueError, match="Invalid AF3 ZIP"):
        import_af3_results(bad, tmp_path / "out")
    with zipfile.ZipFile(bad, "w") as f:
        f.writestr("MODEL.cif", "a")
        f.writestr("model.cif", "b")
    with pytest.raises(ValueError, match="Duplicate AF3"):
        import_af3_results(bad, tmp_path / "out")
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("content", ["not a CIF", "data_empty\n_entry.id empty\n"])
def test_af3_invalid_or_empty_coordinates_are_not_models(tmp_path, content):
    archive = tmp_path / "invalid.zip"
    with zipfile.ZipFile(archive, "w") as f:
        f.writestr("invalid_model.cif", content)
    result = import_af3_results(archive, tmp_path / "out")
    assert result["status"] == "no_models_found"
    assert not result["models"] and result["model_errors"]


def test_local_af3_process_without_models_is_failure(tmp_path):
    job = tmp_path / "job.json"
    job.write_text(json.dumps({"dialect": "alphafold3", "sequences": [{"protein": {}}]}))
    fake = tmp_path / "not_an_af3_runner.py"
    fake.write_text('print("fixture exits without prediction")\n')
    result = run_local_af3(
        job,
        tmp_path / "run",
        python_executable=sys.executable,
        run_script=fake,
        model_dir=tmp_path,
        database_dir=tmp_path,
        timeout_seconds=10,
    )
    assert result["status"] == "failed" and result["returncode"] == 0
    assert result["model_files"] == []
