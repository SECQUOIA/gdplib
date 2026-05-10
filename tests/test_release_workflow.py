from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
AGENTS = ROOT / "AGENTS.md"
PUBLISH_WORKFLOW = ROOT / ".github" / "workflows" / "publish.yml"
README = ROOT / "README.md"
PIXI = ROOT / "pixi.toml"


def test_publish_workflow_builds_and_validates_before_publishing():
    workflow = PUBLISH_WORKFLOW.read_text()

    build_index = workflow.index("python -m build")
    check_index = workflow.index("python -m twine check dist/*")
    install_index = workflow.index("python -m pip install dist/*.whl")
    publish_index = workflow.index("pypa/gh-action-pypi-publish@release/v1")

    assert build_index < check_index < install_index < publish_index
    assert "fetch-depth: 0" in workflow
    assert "python-package-distributions" in workflow
    assert "id-token: write" in workflow
    assert "name: pypi" in workflow


def test_publish_workflow_uses_release_trigger_and_supported_python_matrix():
    workflow = PUBLISH_WORKFLOW.read_text()

    assert "\non:\n  release:\n    types: [published]\n" in workflow
    assert 'python-version: ["3.10", "3.11", "3.12"]' in workflow


def test_readme_documents_release_process_and_pixi_policy():
    readme = README.read_text()

    for phrase in [
        ".github/workflows/publish.yml",
        "setuptools.build_meta",
        "setuptools_scm",
        "PyPI Trusted Publishing",
        "pixi run test",
        "pixi run lint",
        "python -m twine check dist/*",
        "linux-64",
        "pixi.lock",
        "pixi.toml",
        "macOS and Windows users",
        "pip workflow",
        "default Pixi environment intentionally excludes optional external",
    ]:
        assert phrase in readme


def test_pixi_manifest_matches_documented_platform_policy():
    pixi = PIXI.read_text()
    readme = README.read_text()

    assert 'platforms = ["linux-64"]' in pixi
    assert "The committed Pixi support surface is `linux-64` only" in readme
    assert "pixi install" in readme
    assert "pixi run test" in readme
    assert "pixi run lint" in readme


def test_agents_documents_release_and_pixi_policy_lessons():
    agents = AGENTS.read_text()

    for phrase in [
        "The committed Pixi support surface is exactly the platform list",
        "`pixi.toml` and the matching entries in `pixi.lock`",
        "leave the manifest and lock unchanged",
        "deprecated GraphQL fields",
        "Keep optional external solver stacks and licensed solver bindings out",
        "tests/test_release_workflow.py",
    ]:
        assert phrase in agents
