import asyncio
import json
import sys

import httpx
import pytest

from molquarry import MolQuarryError
from molquarry.cli import main


def test_cli_json_and_error_exit_code(capsys, tmp_path):
    assert main(["--home", str(tmp_path), "sources", "--implemented"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert len(data["sources"]) == 41
    assert main(["query", "surechembl", "search", "--params", "{}"]) == 2
    captured = capsys.readouterr()
    assert not captured.out
    assert json.loads(captured.err)["error"]["code"] == "unknown_operation"
    assert main(["query", "pubchem", "properties", "--params", "[]"]) == 2
    assert json.loads(capsys.readouterr().err)["error"]["code"] == "invalid_parameters"


def test_cli_saved_plan_and_no_overwrite(capsys, tmp_path):
    destination = tmp_path / "plan.json"
    args = [
        "plan",
        "rcsb",
        "structure",
        "--params",
        '{"pdb_id":"7KNW"}',
        "--output",
        str(destination),
    ]
    assert main(args) == 0
    capsys.readouterr()
    original = destination.read_bytes()
    assert json.loads(original)["source"] == "rcsb"
    assert main(args) == 2
    assert destination.read_bytes() == original


def test_bounded_paging_retains_continuation(make_quarry):
    q = make_quarry(
        lambda _: httpx.Response(
            200, json={"total_count": 99, "result_set": [{"identifier": "7KNW"}]}
        )
    )
    pages = list(q.iter_pages("rcsb", "search", query="EGFR", limit=1, max_pages=1))
    assert len(pages) == 1 and pages[0].next_parameters is not None
    with pytest.raises(MolQuarryError):
        list(q.iter_pages("rcsb", "search", query="EGFR", max_pages=0))


def test_repeated_cursor_detected(make_quarry):
    q = make_quarry(
        lambda _: httpx.Response(
            200, json={"studies": [], "totalCount": 10, "nextPageToken": "repeat"}
        )
    )
    with pytest.raises(MolQuarryError) as error:
        list(q.iter_pages("clinicaltrials", "search", query="EGFR", page_token="repeat", limit=1))
    assert error.value.code == "pagination_loop"


def test_mcp_stdio_handshake_discovery_and_structured_error(tmp_path):
    pytest.importorskip("mcp")
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    async def run():
        server = StdioServerParameters(
            command=sys.executable,
            args=["-m", "molquarry.mcp_server"],
            env={"MOLQUARRY_HOME": str(tmp_path), "MOLQUARRY_WORKSPACE": str(tmp_path)},
        )
        async with stdio_client(server) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                tools = await session.list_tools()
                assert {t.name for t in tools.tools} == {
                    "list_sources",
                    "describe_source",
                    "query_database",
                    "plan_download",
                    "download_data",
                    "import_catalog",
                    "local_catalogs",
                    "search_local_catalog",
                    "collect_target_evidence",
                    "build_target_bundle",
                    "import_af3_results",
                    "screen_compound_sourcing",
                    "export_target_inventory",
                }
                reply = await session.call_tool("list_sources", {})
                assert not reply.isError and len(reply.structuredContent["sources"]) == 41
                description = await session.call_tool("describe_source", {"source": "chembl"})
                assert "activities" in description.structuredContent["operations"]
                failure = await session.call_tool(
                    "query_database",
                    {
                        "source": "chembl",
                        "operation": "activities",
                        "parameters": {"target_id": "typo"},
                    },
                )
                assert failure.isError
                assert failure.structuredContent["error"]["code"] == "invalid_parameters"
                plan = await session.call_tool(
                    "plan_download",
                    {"source": "rcsb", "operation": "structure", "parameters": {"pdb_id": "7KNW"}},
                )
                assert plan.structuredContent["url"].endswith("7KNW.cif")
                too_big = {**plan.structuredContent, "estimated_bytes": 999999999}
                rejected = await session.call_tool("download_data", {"plan": too_big})
                assert rejected.isError
                assert rejected.structuredContent["error"]["code"] == "download_too_large"
                (tmp_path / "imports").mkdir()
                (tmp_path / "imports" / "mce.csv").write_text("catalog_id,smiles\nHY-1,CCO\n")
                imported = await session.call_tool(
                    "import_catalog",
                    {
                        "source": "mce",
                        "relative_path": "imports/mce.csv",
                        "source_version": "fixture",
                    },
                )
                assert not imported.isError
                found = await session.call_tool(
                    "search_local_catalog",
                    {
                        "snapshot_id": imported.structuredContent["snapshot_id"],
                        "field": "catalog_id",
                        "value": "HY-1",
                    },
                )
                assert found.structuredContent["returned"] == 1
                rejected_path = await session.call_tool(
                    "import_catalog",
                    {
                        "source": "mce",
                        "relative_path": "../outside.csv",
                        "source_version": "fixture",
                    },
                )
                assert rejected_path.isError
                assert rejected_path.structuredContent["error"]["code"] == "invalid_path"
                for tool, parameters in [
                    (
                        "collect_target_evidence",
                        {"config": {"targets": ["SND1"]}, "output_relative_path": "../escape"},
                    ),
                    (
                        "build_target_bundle",
                        {
                            "dossier_relative_path": "/outside/dossier.json",
                            "curation": {},
                            "output_relative_path": "bundle",
                        },
                    ),
                    (
                        "import_af3_results",
                        {"archive_relative_path": "../archive.zip", "output_relative_path": "af3"},
                    ),
                    (
                        "screen_compound_sourcing",
                        {
                            "input_relative_path": "../shortlist.sdf",
                            "output_relative_path": "screen",
                        },
                    ),
                    (
                        "export_target_inventory",
                        {
                            "dossier_relative_path": "../dossier.json",
                            "output_relative_path": "table",
                        },
                    ),
                ]:
                    failure = await session.call_tool(tool, parameters)
                    assert failure.isError
                    assert failure.structuredContent["error"]["code"] == "invalid_path"
                try:
                    import openpyxl
                    from rdkit import Chem
                except ImportError:
                    return  # Core MCP remains testable without the optional chemistry extra.
                sdf_path = tmp_path / "input.sdf"
                with Chem.SDWriter(str(sdf_path)) as writer:
                    writer.write(Chem.MolFromSmiles("CCO"))
                sourcing = await session.call_tool(
                    "screen_compound_sourcing",
                    {
                        "input_relative_path": "input.sdf",
                        "output_relative_path": "sourcing",
                        "config": {
                            "pubchem": False,
                            "chembl": False,
                            "unichem": False,
                            "max_mcule_queries": 0,
                        },
                    },
                )
                assert not sourcing.isError and sourcing.structuredContent["input_records"] == 1
                dossier = {
                    "chembl_details": {"molecule": [], "document": []},
                    "activities": [],
                    "bindingdb": [],
                    "coverage": [],
                    "papers": [],
                }
                (tmp_path / "dossier.json").write_text(json.dumps(dossier))
                inventory = await session.call_tool(
                    "export_target_inventory",
                    {
                        "dossier_relative_path": "dossier.json",
                        "output_relative_path": "inventory",
                    },
                )
                assert not inventory.isError and inventory.structuredContent["compounds"] == 0
                book = openpyxl.load_workbook(tmp_path / "inventory/ligands.xlsx", read_only=True)
                assert "Measurements" in book.sheetnames
                book.close()

    asyncio.run(run())
