"""Offline regression tests of the live acceptance runner, never live-success evidence."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

from molquarry.catalog import SOURCES

spec = importlib.util.spec_from_file_location(
    "public_acceptance", Path(__file__).parents[1] / "scripts" / "verify_public.py"
)
acceptance = importlib.util.module_from_spec(spec)
spec.loader.exec_module(acceptance)


def test_every_source_has_an_explicit_acceptance_or_deferral():
    credentialed = {s.id for s in SOURCES.values() if s.credential_env}
    assert set(acceptance.FUNCTIONS) | set(acceptance.WEB_CASES) | credentialed == set(SOURCES)
    assert not set(acceptance.FUNCTIONS) & credentialed


def test_drug_name_match_without_original_approval_fails(tmp_path):
    rows = [
        {"application_number": n, "submissions": [], "products": []}
        for n in ["NDA021335", "NDA021588"]
    ]
    check = acceptance.Check("drugsfda", None, tmp_path, None)
    check.query = lambda *a, **kw: rows
    with pytest.raises(AssertionError, match="original approval date"):
        acceptance.drugsfda(check)


def test_public_runner_defers_credentials_even_if_configured(monkeypatch, tmp_path):
    monkeypatch.setenv("REAL_API_KEY", "already-configured")
    monkeypatch.setattr(acceptance, "MolQuarry", lambda **kw: pytest.fail("No network client"))
    row = acceptance.run_source("enamine", tmp_path, SimpleNamespace())
    assert row["status"] == "deferred_credentials"
