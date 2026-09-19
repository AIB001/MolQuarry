"""Bounded, resumable computed conformers for database inventories."""

import json
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path

from .compounds import file_label, prepare_molecule
from .journal import digest


def _conformer(row):
    from rdkit import Chem

    result = {"compound_id": row["compound_id"], "origin": "computed_not_experimental"}
    try:
        molecule = Chem.MolFromSmiles(row["smiles_rdkit"])
        if molecule.GetNumHeavyAtoms() > 120:
            return {**result, "status": "not_prepared_size_budget"}
        prepared, processing = prepare_molecule(molecule)
        if Chem.MolToInchiKey(Chem.RemoveHs(prepared)) != row["inchikey"]:
            raise ValueError("Conformer identity differs from input")
        prepared.SetProp("_Name", row["compound_id"])
        return {
            **result,
            "status": "prepared",
            "processing": processing,
            "inchikey": row["inchikey"],
            "pmids": row["pmids"],
            "molblock": Chem.MolToMolBlock(prepared),
        }
    except (ValueError, RuntimeError) as exc:
        return {**result, "status": "preparation_error", "error": str(exc)}


def export_inventory_conformers(inventory_path, output_dir, *, workers=2, progress=None):
    """Generate labeled 3D files, reusing only a matching immutable input request."""
    from rdkit import Chem, rdBase

    if not 1 <= workers <= 4:
        raise ValueError("workers must be between 1 and 4")
    source = json.loads(Path(inventory_path).read_text())
    rows = [r for r in source["compounds"] if r["chemical_status"] == "parsed"]
    spec = {
        "identities": [
            {k: r[k] for k in ["compound_id", "smiles_rdkit", "inchikey", "pmids"]} for r in rows
        ],
        "rdkit": rdBase.rdkitVersion,
        "processing": "ETKDGv3 seed=20260919, timeout=10s; MMFF94/UFF, 1000 iterations",
    }
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    request = root / "request.json"
    if request.exists():
        if json.loads(request.read_text()) != spec:
            raise ValueError("Conformer request differs from existing run")
    elif any(root.iterdir()):
        raise ValueError("Conformer directory already contains unrelated files")
    else:
        request.write_text(json.dumps(spec, indent=2) + "\n")
    pending = [r for r in rows if not (root / f"{file_label(r['compound_id'])}.json").exists()]
    progress = progress or (lambda message: None)
    with ProcessPoolExecutor(max_workers=workers, mp_context=get_context("spawn")) as pool:
        for index, result in enumerate(pool.map(_conformer, pending), 1):
            label = file_label(result["compound_id"])
            if result["status"] == "prepared":
                molecule = Chem.MolFromMolBlock(result.pop("molblock"), removeHs=False)
                if molecule is None or not molecule.GetConformer().Is3D():
                    result.update(status="serialization_error")
                else:
                    for key in ["origin", "inchikey", "pmids", "processing"]:
                        molecule.SetProp(key, json.dumps(result[key]))
                    with Chem.SDWriter(str(root / f"{label}.sdf")) as writer:
                        writer.write(molecule)
                    result.update(status="written", file=f"{label}.sdf")
            record = root / f"{label}.json"
            temporary = record.with_suffix(".part")
            temporary.write_text(json.dumps(result, indent=2) + "\n")
            temporary.replace(record)
            if index % 25 == 0 or index == len(pending):
                progress(f"Computed conformers: {index}/{len(pending)}")
    results = [
        json.loads((root / f"{file_label(r['compound_id'])}.json").read_text()) for r in rows
    ]
    result = {
        "request_sha256": digest(spec),
        "records": results,
        "attempted": len(rows),
        "written": sum(r["status"] == "written" for r in results),
        "nonconverged": sum(r.get("processing", {}).get("converged") is False for r in results),
    }
    (root / "index.json").write_text(json.dumps(result, indent=2) + "\n")
    return result
