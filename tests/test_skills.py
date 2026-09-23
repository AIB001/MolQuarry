import json
import re
from pathlib import Path

import pytest

from molquarry import MolQuarryError
from molquarry.cli import main
from molquarry.skills import list_skills, read_skill


def test_skill_discovery_and_every_resource_is_readable():
    rows = list_skills()["skills"]
    assert len(rows) >= 6
    assert len({row["name"] for row in rows}) == len(rows)
    for row in rows:
        assert row["description"]
        assert row["entrypoint"] in row["resources"]
        for path in row["resources"]:
            result = read_skill(row["name"], path)
            assert result["content"] and result["path"] == path
        text = read_skill(row["name"])["content"]
        for linked in re.findall(r"\]\(((?:references|agents)/[^)]+)\)", text):
            assert linked in row["resources"]


@pytest.mark.parametrize(
    ("name", "path", "code"),
    [
        ("../secrets", "SKILL.md", "invalid_path"),
        ("molquarry-analogue-search", "../secrets", "invalid_path"),
        ("molquarry-analogue-search", "/etc/passwd", "invalid_path"),
        ("molquarry-analogue-search", "references\\secrets", "invalid_path"),
        ("missing-skill", "SKILL.md", "unknown_skill"),
        ("molquarry-analogue-search", "not-a-resource.txt", "unknown_skill_resource"),
    ],
)
def test_only_listed_skill_resources_can_be_read(name, path, code):
    with pytest.raises(MolQuarryError) as exc:
        read_skill(name, path)
    assert exc.value.code == code


def test_new_skill_tool_examples_validate_against_real_schemas(make_quarry):
    quarry = make_quarry(lambda _: pytest.fail("Skill schema validation must stay offline"))
    count = 0
    for row in list_skills()["skills"]:
        text = read_skill(row["name"])["content"]
        for block in re.findall(r"```json\n(.*?)\n```", text, flags=re.DOTALL):
            payload = json.loads(block)
            if not {"source", "operation", "parameters"} <= payload.keys():
                continue  # Existing curation JSON is not a query_database payload.
            provider = quarry.providers[payload["source"]]
            provider.operations[payload["operation"]].input_model.model_validate(
                payload["parameters"]
            )
            count += 1
    assert count >= 6


def test_cli_skill_entrypoint_and_reference(capsys, tmp_path):
    args = ["--home", str(tmp_path)]
    assert main([*args, "skills"]) == 0
    rows = json.loads(capsys.readouterr().out)["skills"]
    assert any(row["name"] == "molquarry-structure-templates" for row in rows)
    assert (
        main([*args, "skill", "molquarry-target-modulators", "--path", "references/workflow.md"])
        == 0
    )
    assert json.loads(capsys.readouterr().out)["content"]


def test_runtime_and_project_versions_agree():
    import tomllib

    from molquarry import __version__

    project = Path(__file__).resolve().parents[1] / "pyproject.toml"
    assert tomllib.loads(project.read_text())["project"]["version"] == __version__
