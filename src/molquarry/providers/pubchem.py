from typing import Annotated, Literal
from urllib.parse import quote

from pydantic import Field

from ..models import InputModel
from .base import Operation, Page, Provider

BASE = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"
PROPERTIES = (
    "MolecularFormula,MolecularWeight,SMILES,ConnectivitySMILES,InChI,InChIKey,"
    "XLogP,TPSA,HBondDonorCount,HBondAcceptorCount"
)


class CompoundInput(InputModel):
    identifier: str = Field(min_length=1, max_length=4000)
    namespace: Literal["name", "cid", "inchikey", "smiles"] = "name"


class SimilarityInput(InputModel):
    smiles: str = Field(min_length=1, max_length=4000)
    threshold: int = Field(default=90, ge=0, le=100)
    limit: int = Field(default=20, ge=1, le=100)


class AssayInput(InputModel):
    aid: int = Field(gt=0)


class SDFInput(InputModel):
    cids: list[Annotated[int, Field(gt=0)]] = Field(min_length=1, max_length=100)
    record_type: Literal["2d", "3d"] = "2d"


class PubChem(Provider):
    id = "pubchem"
    download_hosts = frozenset({"pubchem.ncbi.nlm.nih.gov"})
    operations = {
        "properties": Operation(
            "Resolve a compound and retrieve identity/properties; SMILES uses POST.",
            CompoundInput,
            {"identifier": "aspirin"},
        ),
        "similarity": Operation(
            "PubChem 2D fingerprint similarity (not a patent novelty verdict).",
            SimilarityInput,
            {"smiles": "CC(=O)Oc1ccccc1C(=O)O", "limit": 5},
        ),
        "assay": Operation(
            "BioAssay description, readout definitions and target metadata by AID.",
            AssayInput,
            {"aid": 1},
        ),
    }
    downloads = {
        "sdf": Operation(
            "Download up to 100 known CIDs as 2D/3D SDF.",
            SDFInput,
            {"cids": [2244], "record_type": "2d"},
        )
    }

    def query(self, operation, params):
        if operation == "properties":
            if params.namespace == "smiles":
                response = self.request(
                    "POST",
                    f"{BASE}/compound/smiles/property/{PROPERTIES}/JSON",
                    form={"smiles": params.identifier},
                )
            else:
                identifier = quote(params.identifier, safe="")
                response = self.request(
                    "GET",
                    f"{BASE}/compound/{params.namespace}/{identifier}/property/{PROPERTIES}/JSON",
                )
            return Page(response.data["PropertyTable"]["Properties"], [response.provenance])
        if operation == "assay":
            response = self.request("GET", f"{BASE}/assay/aid/{params.aid}/description/JSON")
            return Page(response.data["PC_AssayContainer"], [response.provenance])
        response = self.request(
            "POST",
            f"{BASE}/compound/fastsimilarity_2d/smiles/cids/JSON",
            params={"Threshold": params.threshold, "MaxRecords": params.limit},
            form={"smiles": params.smiles},
        )
        rows = [{"CID": cid} for cid in response.data["IdentifierList"]["CID"]]
        warnings = ["Results are capped by MaxRecords; no exhaustive similarity search is implied."]
        return Page(rows, [response.provenance], warnings=warnings)

    def plan(self, operation, params):
        ids = ",".join(map(str, params.cids))
        return self.make_plan(
            operation,
            params,
            url=f"{BASE}/compound/cid/{ids}/SDF?record_type={params.record_type}",
            filename=f"pubchem_{params.cids[0]}_{len(params.cids)}_{params.record_type}.sdf",
            format="sdf",
        )
