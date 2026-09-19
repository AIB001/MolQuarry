import gzip
import hashlib
import json
from pathlib import Path

import httpx
import pytest

from molquarry import MolQuarryError

FASTA = b">sp|P00533|EGFR_HUMAN\nMRPSGT\n"


def stream_response(data=FASTA, **kwargs):
    return httpx.Response(200, stream=httpx.ByteStream(data), **kwargs)


def test_stream_manifest_and_no_overwrite(make_quarry, tmp_path):
    q = make_quarry(
        lambda _: stream_response(headers={"Content-Length": str(len(FASTA)), "ETag": "revision"})
    )
    plan = q.plan_download("uniprot", "fasta", accession="P00533")
    plan.expected_sha256 = hashlib.sha256(FASTA).hexdigest()
    result = q.download(plan)
    assert Path(result.path).read_bytes() == FASTA
    manifest = json.loads(Path(result.manifest_path).read_text())
    assert manifest["files"][0]["sha256"] == result.sha256 == plan.expected_sha256
    assert manifest["upstream_checksum_verified"] is True
    assert manifest["plan"]["parameters"] == {"accession": "P00533"}
    assert (
        manifest["source_version"] is None
    )  # Retrieval date is never invented as a source release.
    assert not list(tmp_path.rglob(".molquarry-*"))
    with pytest.raises(MolQuarryError) as error:
        q.download(plan)
    assert error.value.code == "file_exists"
    assert Path(result.path).read_bytes() == FASTA


def test_signed_download_credentials_are_not_saved_in_nested_plan(make_quarry):
    seen = []
    q = make_quarry(lambda r: seen.append(str(r.url)) or stream_response())
    plan = q.plan_download("uniprot", "fasta", accession="P00533")
    plan.url += "?X-Amz-Signature=temporary-secret&generation=123"
    plan.notes = [plan.url]
    result = q.download(plan)
    assert "temporary-secret" in seen[0]  # still sent to the intended source
    assert "temporary-secret" not in json.dumps(result.manifest)
    assert "generation=123" in result.manifest["plan"]["url"]
    assert "REDACTED" in result.manifest["plan"]["notes"][0]


def test_official_vendor_login_redirect_is_reported_without_following(make_quarry):
    calls = []

    def handle(request):
        calls.append(str(request.url))
        if "download.php" in request.url.path:
            return httpx.Response(302, headers={"location": "https://shop.lifechemicals.com/login"})
        return httpx.Response(200, text='<a href="https://lifechemicals.com/data.zip">SDF</a>')

    q = make_quarry(handle)
    plan = q.plan_download("lifechemicals", "artifact", url="https://lifechemicals.com/data.zip")
    plan.url = "https://lifechemicals.com/download.php?path=data.zip"
    with pytest.raises(MolQuarryError) as exc:
        q.download(plan)
    assert exc.value.code == "authentication_required"
    assert exc.value.details["login_url"] == "https://shop.lifechemicals.com/login"
    assert not any("shop.lifechemicals.com" in url for url in calls)


@pytest.mark.parametrize(
    ("change", "code"),
    [
        ({"filename": "../../secret"}, "invalid_filename"),
        ({"filename": "C:\\secret"}, "invalid_filename"),
        ({"url": "http://rest.uniprot.org/file"}, "invalid_download_url"),
        ({"url": "https://127.0.0.1/private"}, "invalid_download_url"),
        ({"url": "https://rest.uniprot.org.evil.test/file"}, "invalid_download_url"),
        ({"url": "https://user:secret@rest.uniprot.org/file"}, "invalid_download_url"),
        ({"estimated_bytes": 1000000000}, "download_too_large"),
        ({"expected_sha256": "not-a-hash"}, "invalid_checksum"),
    ],
)
def test_invalid_plan_never_opens_network(make_quarry, change, code):
    def handler(_):
        raise AssertionError("No request expected")

    q = make_quarry(handler)
    plan = q.plan_download("uniprot", "fasta", accession="P00533").model_copy(update=change)
    with pytest.raises(MolQuarryError) as error:
        q.download(plan)
    assert error.value.code == code


@pytest.mark.parametrize(
    ("response", "changes", "budget", "code"),
    [
        (lambda: stream_response(), {"expected_sha256": "a" * 64}, 1000, "checksum_mismatch"),
        (lambda: stream_response(), {}, 5, "download_too_large"),
        (
            lambda: stream_response(headers={"Content-Length": "9999"}),
            {},
            1000,
            "download_too_large",
        ),
        (
            lambda: stream_response(headers={"Content-Length": "999"}),
            {},
            1000,
            "incomplete_download",
        ),
        (lambda: stream_response(b""), {}, 1000, "incomplete_download"),
        (
            lambda: stream_response(b"<html>error</html>", headers={"Content-Type": "text/html"}),
            {},
            1000,
            "invalid_response",
        ),
        (
            lambda: httpx.Response(302, headers={"Location": "https://localhost/secret"}),
            {},
            1000,
            "invalid_download_url",
        ),
        (lambda: httpx.Response(206, stream=httpx.ByteStream(FASTA)), {}, 1000, "invalid_response"),
    ],
)
def test_failed_download_never_publishes_artifact(
    make_quarry, tmp_path, response, changes, budget, code
):
    q = make_quarry(lambda _: response())
    plan = q.plan_download("uniprot", "fasta", accession="P00533").model_copy(update=changes)
    with pytest.raises(MolQuarryError) as error:
        q.download(plan, output_dir=tmp_path / "files", max_bytes=budget)
    assert error.value.code == code
    assert list((tmp_path / "files").iterdir()) == []


def test_stream_interruption_removes_partial_file(make_quarry, tmp_path):
    class Broken(httpx.SyncByteStream):
        def __iter__(self):
            yield b"partial"
            raise httpx.ReadError("dropped connection")

    q = make_quarry(lambda _: httpx.Response(200, stream=Broken()))
    plan = q.plan_download("uniprot", "fasta", accession="P00533")
    with pytest.raises(MolQuarryError) as error:
        q.download(plan, output_dir=tmp_path / "files")
    assert error.value.code == "download_interrupted"
    assert list((tmp_path / "files").iterdir()) == []


def test_static_download_urls_and_validation(make_quarry):
    q = make_quarry(lambda _: pytest.fail("No request needed to construct static artifact URLs"))
    assert q.plan_download("rcsb", "structure", pdb_id="7knw").url.endswith("7KNW.cif")
    assert q.plan_download("chembl", "sdf", chembl_id="CHEMBL25").url.endswith("CHEMBL25.sdf")
    assert "/2244,2519/SDF" in q.plan_download("pubchem", "sdf", cids=[2244, 2519]).url
    with pytest.raises(MolQuarryError):
        q.plan_download("pubchem", "sdf", cids=[0])


def test_error_body_is_not_published_as_a_sequence(make_quarry, tmp_path):
    q = make_quarry(lambda _: stream_response(b'{"error":"down for maintenance"}'))
    plan = q.plan_download("uniprot", "fasta", accession="P00533")
    with pytest.raises(MolQuarryError) as error:
        q.download(plan, output_dir=tmp_path / "files")
    assert error.value.code == "unexpected_format"
    assert list((tmp_path / "files").iterdir()) == []


def test_bulk_catalog_download_preserves_compressed_bytes_and_source_checksum(make_quarry):
    original = b"SMILES\tID\nCCO\tMCULE-1\n"
    compressed = gzip.compress(original, mtime=0)
    checksum = hashlib.sha256(compressed).hexdigest()

    def handler(request):
        if request.url.host == "mcule.com":
            return httpx.Response(
                200,
                json={
                    "id": 1,
                    "last_updated": "2026-08-01",
                    "files": [
                        {
                            "file_type": "smi.gz",
                            "filename": "fixture.smi.gz",
                            "download_url": "https://dl.mcule.com/database/fixture.smi.gz",
                            "sha256_checksum": checksum,
                            "size_mb": 0.001,
                        }
                    ],
                },
            )
        return stream_response(compressed, headers={"Content-Length": str(len(compressed))})

    q = make_quarry(handler)
    artifact = q.download(q.plan_download("mcule", "dataset", dataset_id=1), max_bytes=1000)
    assert artifact.sha256 == checksum
    assert gzip.decompress(Path(artifact.path).read_bytes()) == original
    assert artifact.manifest["upstream_checksum_verified"] is True
    assert artifact.manifest["source_version"] == "2026-08-01"
