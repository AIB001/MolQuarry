import gzip
import hashlib
import json
from pathlib import Path

import httpx
import pytest

from molquarry import MolQuarryError
from molquarry.downloads import validate_signature


def test_official_file_discovery_pagination_and_exact_membership(make_quarry):
    html = """<html><a href="../">parent</a><a href="2026-09/">release</a>
    <a href="molecules.parquet">data</a><a href="https://evil.test/steal.csv">bad</a>
    <a href="?C=N;O=D">sort</a><a href="https://ftp.ebi.ac.uk/other/">outside root</a></html>"""
    q = make_quarry(lambda _: httpx.Response(200, text=html))
    page = q.query("surechembl", "files", limit=1)
    assert page.total == 2 and page.records[0]["kind"] == "directory"
    second = q.query("surechembl", "files", **page.next_parameters)
    row = second.records[0]
    plan = q.plan_download("surechembl", "artifact", url=row["url"])
    assert plan.format == "parquet" and plan.provenance
    with pytest.raises(MolQuarryError) as error:
        q.plan_download("surechembl", "artifact", url=row["url"].replace("molecules", "unlisted"))
    assert error.value.code == "artifact_not_listed"


@pytest.mark.parametrize(
    "path", ["../", "%2e%2e/", "/etc/", "a/../../", "https://x/", "a\\b/", "foo"]
)
def test_bulk_traversal_rejected_before_request(make_quarry, path):
    q = make_quarry(lambda _: pytest.fail("Invalid path must not access network"))
    with pytest.raises(MolQuarryError) as error:
        q.query("zinc", "files", path=path)
    assert error.value.code == "invalid_parameters"


def test_login_page_is_not_an_empty_successful_catalog(make_quarry):
    q = make_quarry(lambda _: httpx.Response(200, text="<html>Please log in</html>"))
    with pytest.raises(MolQuarryError) as error:
        q.query("biolip", "files")
    assert error.value.code == "no_download_links"


def test_bindingdb_download_link_and_lifechemicals_query_filename(make_quarry):
    def handle(request):
        html = (
            '<a href="/rwd/bind/chemsearch/marvin/SDFdownload.jsp?download_file='
            '/rwd/bind/downloads/BindingDB_All_202609_tsv.zip">file</a>'
            if request.url.host == "www.bindingdb.org"
            else '<a href="/download.php?path=%2FScreening%20Libraries%2Flibrary.zip">file</a>'
        )
        return httpx.Response(200, text=html)

    q = make_quarry(handle)
    b = q.query("bindingdb", "files").records[0]
    assert b["url"].endswith("/downloads/BindingDB_All_202609_tsv.zip")
    life = q.query("lifechemicals", "files").records[0]
    assert life["filename"] == "library.zip" and "download.php?path=" in life["url"]


def test_ord_download_pins_revision_and_lfs_sha256(make_quarry):
    sha = "a" * 40
    content = b"PAR1fixturePAR1"
    checksum = hashlib.sha256(content).hexdigest()
    path = "data/00/ord_dataset-fixture.parquet"

    def handle(request):
        if "/revision/" in request.url.path:
            return httpx.Response(200, json={"sha": sha})
        if "/paths-info/" in request.url.path:
            assert json.loads(request.content) == {"paths": [path]}
            return httpx.Response(
                200,
                json=[
                    {"path": path, "type": "file", "size": len(content), "lfs": {"oid": checksum}}
                ],
            )
        assert f"/resolve/{sha}/" in request.url.path
        return httpx.Response(200, stream=httpx.ByteStream(content))

    q = make_quarry(handle)
    plan = q.plan_download("ord", "dataset", path=path)
    assert plan.expected_sha256 == checksum and plan.source_version == sha
    result = q.download(plan)
    assert result.manifest["upstream_checksum_verified"]


def test_tdc_never_executes_metadata_and_rejects_restricted_files(make_quarry, tmp_path):
    marker = tmp_path / "should-not-exist"
    metadata = (
        f"open({str(marker)!r}, 'w').write('unsafe')\n"
        "name2id = {'fixture': 42}\nname2type = {'fixture': 'tab'}\n"
    )

    def handle(request):
        if request.url.host == "raw.githubusercontent.com":
            return httpx.Response(200, text=metadata)
        return httpx.Response(200, json={"label": "fixture.tab", "restricted": True})

    q = make_quarry(handle)
    page = q.query("tdc", "datasets", query="fixture")
    assert page.records[0]["file_id"] == 42 and not marker.exists()
    with pytest.raises(MolQuarryError) as error:
        q.plan_download("tdc", "dataset", name="fixture")
    assert error.value.code == "access_denied"


def test_deposit_source_validation_for_shared_figshare_host(make_quarry):
    q = make_quarry(lambda _: httpx.Response(200, json={"authors": [{"id": 1}], "files": []}))
    with pytest.raises(MolQuarryError) as error:
        q.query("depmap", "release", article_id=1)
    assert error.value.code == "unverified_source"


@pytest.mark.parametrize(
    ("fmt", "content"),
    [
        ("zip", b"PK\x03\x04content"),
        ("parquet", b"PAR1payloadPAR1"),
        ("tsv", b"id\tsmiles\n1\tCCO\n"),
        ("tsv.gz", gzip.compress(b"id\tsmiles\n")),
        ("xml", b'<?xml version="1.0"?><root/>'),
    ],
)
def test_bulk_container_checks(fmt, content, tmp_path):
    path = tmp_path / "data"
    path.write_bytes(content)
    validate_signature(path, fmt)
    path.write_bytes(b"<!DOCTYPE html><html>sign in</html>")
    with pytest.raises(MolQuarryError):
        validate_signature(path, fmt)


def test_local_snapshot_exact_lookup_paging_and_original_retention(make_quarry, tmp_path):
    raw = "\ufeffcatalog_id,smiles,name\nHY-1,CCO,乙醇\nHY-2,CCCO,propanol\n"
    path = tmp_path / "vendor.csv"
    path.write_text(raw, encoding="utf-8")
    q = make_quarry(lambda _: pytest.fail("Local import/search cannot use network"))
    result = q.import_catalog("mce", path, source_version="vendor-export-2026-09")
    assert result["records"] == 2
    assert Path(result["path"]).read_bytes() == path.read_bytes()
    path.write_text("changed after import")
    page = q.local_search(snapshot_id=result["snapshot_id"], field="catalog_id", value="HY-1")
    assert page.records[0]["fields"]["name"] == "乙醇" and page.total == 1
    page = q.local_search(snapshot_id=result["snapshot_id"], limit=1)
    assert page.next_parameters and page.total == 2
    assert q.local_search(**page.next_parameters).next_parameters is None
    assert not q.local_search(
        snapshot_id=result["snapshot_id"], field="catalog_id", value="' OR 1=1 --"
    ).records
    assert q.local_catalogs()["catalogs"][0]["sha256"] == result["sha256"]


def test_import_limits_rollback_and_gzip_budget(make_quarry, tmp_path):
    path = tmp_path / "vendor.tsv.gz"
    path.write_bytes(gzip.compress(b"id\tsmiles\n1\tCCO\n2\tCCC\n"))
    q = make_quarry(lambda _: pytest.fail("No network"))
    with pytest.raises(MolQuarryError) as error:
        q.import_catalog("bld", path, source_version="v1", max_records=1)
    assert error.value.code == "import_too_large"
    assert q.local_catalogs()["catalogs"] == []
    assert list((tmp_path / "catalogs").iterdir()) == []
    path.write_bytes(gzip.compress(b"id\tsmiles\n1\t" + b"C" * 500))
    with pytest.raises(MolQuarryError):
        q.import_catalog("bld", path, source_version="v1", max_bytes=100)
    assert q.local_catalogs()["catalogs"] == []


def test_local_sdf_properties_and_missing_delimiter(make_quarry, tmp_path):
    sdf = (
        "Fixture\nMolQuarry\n\n  0  0  0  0  0  0            999 V2000\nM  END\n"
        "> <Catalog ID>\nBB-1\n\n> <SMILES>\nCCO\n\n"
    )
    path = tmp_path / "catalog.sdf"
    path.write_text(sdf)
    q = make_quarry(lambda _: pytest.fail("No network"))
    imported = q.import_catalog("pdbbind", path, source_version="authorized-fixture")
    row = q.local_search(
        snapshot_id=imported["snapshot_id"], field="Catalog ID", value="BB-1"
    ).records[0]
    assert row["fields"]["_sdf"] == sdf
    repeat = q.import_catalog("pdbbind", path, source_version="authorized-fixture")
    assert repeat["already_imported"] is True
