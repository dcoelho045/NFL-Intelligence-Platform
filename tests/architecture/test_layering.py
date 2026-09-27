"""Architecture tests.

Several acceptance criteria -- "raw data access is separated from business
logic", "numerical claims come from tools", "no RAG/MCP/multi-agent
functionality yet" -- are properties of the codebase, not of any single
function. These tests assert them mechanically so they cannot quietly stop being
true.

The checks read import statements from the AST rather than importing modules, so
a violation is caught even in a branch that never executes.
"""

from __future__ import annotations

import ast
import re
import tomllib
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[2] / "src" / "nfl_platform"

# Packages that must not appear anywhere in src/ until their milestone arrives.
FORBIDDEN_PACKAGES = frozenset(
    {
        "anthropic",
        "langchain",
        "langgraph",
        "llama_index",
        "transformers",
        "sentence_transformers",
        "chromadb",
        "qdrant_client",
        "pinecone",
        "mcp",
        "sklearn",
        "xgboost",
        "lightgbm",
        "torch",
        "tensorflow",
    }
)

# Milestone 3 introduces exactly one LLM SDK, confined to one package. Keeping it
# quarantined is what lets the rest of src/ stay provably free of model calls and
# keeps swapping providers a single-directory change.
PROVIDER_ONLY_PACKAGES = frozenset({"openai"})
PROVIDER_PACKAGE = "llm/providers"


def _module_files(package: str) -> list[Path]:
    return sorted((SRC / package).rglob("*.py"))


def _imported_roots(path: Path) -> set[str]:
    """Top-level package names imported by a module."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            roots.add(node.module.split(".")[0])
    return roots


def _imported_modules(path: Path) -> set[str]:
    """Fully qualified module names imported by a module."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            modules.add(node.module)
    return modules


class TestDetectorItself:
    """A guard that cannot fail is worthless, so prove the detector detects."""

    def test_detects_plain_imports(self, tmp_path: Path):
        sample = tmp_path / "sample.py"
        sample.write_text("import langchain\nimport os\n", encoding="utf-8")
        assert _imported_roots(sample) & FORBIDDEN_PACKAGES == {"langchain"}

    def test_detects_provider_only_imports(self, tmp_path: Path):
        sample = tmp_path / "sample.py"
        sample.write_text("from openai import OpenAI\n", encoding="utf-8")
        assert _imported_roots(sample) & PROVIDER_ONLY_PACKAGES == {"openai"}

    def test_detects_from_imports(self, tmp_path: Path):
        sample = tmp_path / "sample.py"
        sample.write_text("from langgraph.graph import StateGraph\n", encoding="utf-8")
        assert _imported_roots(sample) & FORBIDDEN_PACKAGES == {"langgraph"}

    def test_detects_imports_nested_inside_functions(self, tmp_path: Path):
        """Deferred imports inside a function body must not slip past."""
        sample = tmp_path / "sample.py"
        sample.write_text("def f():\n    import torch\n    return torch\n", encoding="utf-8")
        assert _imported_roots(sample) & FORBIDDEN_PACKAGES == {"torch"}

    def test_records_fully_qualified_module_names(self, tmp_path: Path):
        sample = tmp_path / "sample.py"
        sample.write_text("from nfl_platform.services.base import X\n", encoding="utf-8")
        assert "nfl_platform.services.base" in _imported_modules(sample)

    def test_ignores_unrelated_imports(self, tmp_path: Path):
        sample = tmp_path / "sample.py"
        sample.write_text("import polars as pl\nfrom pathlib import Path\n", encoding="utf-8")
        assert not _imported_roots(sample) & FORBIDDEN_PACKAGES


class TestNoPrematureDependencies:
    """Acceptance criterion: no RAG/MCP/multi-agent/ML functionality yet."""

    @pytest.mark.parametrize("path", _module_files(""), ids=lambda p: p.name)
    def test_module_imports_no_forbidden_package(self, path: Path):
        offenders = _imported_roots(path) & FORBIDDEN_PACKAGES
        assert not offenders, f"{path.name} imports {sorted(offenders)}"

    def test_no_future_milestone_dependency_is_declared(self):
        """Parsed, not grepped: a comment mentioning MCP is not a dependency on it."""
        pyproject = tomllib.loads((SRC.parents[1] / "pyproject.toml").read_text(encoding="utf-8"))
        project = pyproject["project"]
        declared = list(project.get("dependencies", []))
        for group in project.get("optional-dependencies", {}).values():
            declared.extend(group)
        roots = {re.split(r"[<>=!~\[; ]", entry, maxsplit=1)[0].lower() for entry in declared}
        assert not roots & FORBIDDEN_PACKAGES

    def test_exactly_one_llm_sdk_is_declared(self):
        pyproject = tomllib.loads((SRC.parents[1] / "pyproject.toml").read_text(encoding="utf-8"))
        roots = {
            re.split(r"[<>=!~\[; ]", entry, maxsplit=1)[0].lower()
            for entry in pyproject["project"]["dependencies"]
        }
        assert roots & PROVIDER_ONLY_PACKAGES == {"openai"}


class TestLlmSdkIsQuarantined:
    """The model SDK lives in one package; nothing else may call a model."""

    @pytest.mark.parametrize("path", _module_files(""), ids=lambda p: p.name)
    def test_only_the_provider_package_imports_an_llm_sdk(self, path: Path):
        offenders = _imported_roots(path) & PROVIDER_ONLY_PACKAGES
        if PROVIDER_PACKAGE in path.as_posix():
            return
        assert not offenders, (
            f"{path.relative_to(SRC).as_posix()} imports {sorted(offenders)}; "
            f"LLM SDK calls belong in {PROVIDER_PACKAGE}/"
        )

    def test_the_provider_package_actually_exists(self):
        """Guard against the rule passing vacuously if the package is renamed."""
        assert (SRC / "llm" / "providers").is_dir()

    @pytest.mark.parametrize("path", _module_files("analytics"), ids=lambda p: p.name)
    def test_analytics_never_imports_the_llm_layer(self, path: Path):
        offenders = {m for m in _imported_modules(path) if m.startswith("nfl_platform.llm")}
        assert not offenders, f"analytics/{path.name} imports the LLM layer"

    @pytest.mark.parametrize("path", _module_files("services"), ids=lambda p: p.name)
    def test_services_never_import_the_llm_layer(self, path: Path):
        """The deterministic core must stay usable with no model present."""
        offenders = {m for m in _imported_modules(path) if m.startswith("nfl_platform.llm")}
        assert not offenders, f"services/{path.name} imports the LLM layer"


class TestLayerDependencyDirection:
    """Acceptance criterion: raw data access is separated from business logic.

    Dependencies must flow strictly downward:
    api -> services -> analytics -> data -> (domain, config)
    """

    @pytest.mark.parametrize("path", _module_files("analytics"), ids=lambda p: p.name)
    def test_analytics_does_not_depend_on_services_or_api(self, path: Path):
        modules = _imported_modules(path)
        offenders = {
            module
            for module in modules
            if module.startswith(("nfl_platform.services", "nfl_platform.api"))
        }
        assert not offenders, f"analytics/{path.name} imports upward: {sorted(offenders)}"

    @pytest.mark.parametrize("path", _module_files("data"), ids=lambda p: p.name)
    def test_data_layer_does_not_depend_on_anything_above_it(self, path: Path):
        modules = _imported_modules(path)
        offenders = {
            module
            for module in modules
            if module.startswith(
                ("nfl_platform.services", "nfl_platform.api", "nfl_platform.analytics")
            )
        }
        assert not offenders, f"data/{path.name} imports upward: {sorted(offenders)}"

    @pytest.mark.parametrize("path", _module_files("services"), ids=lambda p: p.name)
    def test_services_do_not_depend_on_the_web_framework(self, path: Path):
        """Services must stay importable by MCP and LLM tools without FastAPI."""
        offenders = _imported_roots(path) & {"fastapi", "starlette", "uvicorn", "flask"}
        assert not offenders, f"services/{path.name} imports {sorted(offenders)}"

    @pytest.mark.parametrize("path", _module_files("schemas"), ids=lambda p: p.name)
    def test_schemas_depend_on_neither_the_web_framework_nor_polars(self, path: Path):
        """The DTO contract must be importable standalone by any adapter."""
        offenders = _imported_roots(path) & {"fastapi", "starlette", "polars"}
        assert not offenders, f"schemas/{path.name} imports {sorted(offenders)}"

    @pytest.mark.parametrize("path", _module_files("api"), ids=lambda p: p.name)
    def test_api_does_not_reach_past_the_service_layer(self, path: Path):
        """Routers must not call the data layer or analytics functions directly."""
        modules = _imported_modules(path)
        offenders = {
            module
            for module in modules
            if module.startswith(("nfl_platform.analytics", "nfl_platform.data"))
        }
        assert not offenders, f"api/{path.name} bypasses the service layer: {sorted(offenders)}"

    @pytest.mark.parametrize("path", _module_files("llm"), ids=lambda p: p.name)
    def test_llm_does_not_reach_past_the_service_layer(self, path: Path):
        """Tools must call services, never recompute from frames themselves."""
        modules = _imported_modules(path)
        offenders = {
            module
            for module in modules
            if module.startswith(("nfl_platform.analytics", "nfl_platform.data"))
        }
        assert not offenders, f"llm/{path.name} bypasses the service layer: {sorted(offenders)}"

    @pytest.mark.parametrize("path", _module_files("llm"), ids=lambda p: p.name)
    def test_llm_does_not_depend_on_the_api_adapter(self, path: Path):
        """api -> llm is allowed; llm -> api would make two adapters into a stack."""
        offenders = {m for m in _imported_modules(path) if m.startswith("nfl_platform.api")}
        assert not offenders, f"llm/{path.name} imports the API adapter: {sorted(offenders)}"

    @pytest.mark.parametrize("path", _module_files("llm"), ids=lambda p: p.name)
    def test_llm_never_imports_polars_or_the_web_framework(self, path: Path):
        offenders = _imported_roots(path) & {"polars", "fastapi", "starlette", "uvicorn"}
        assert not offenders, f"llm/{path.name} imports {sorted(offenders)}"


class TestDataFramesStopAtTheServiceBoundary:
    """Polars must not leak above the service layer."""

    @pytest.mark.parametrize("path", _module_files("api"), ids=lambda p: p.name)
    def test_api_never_imports_polars(self, path: Path):
        assert "polars" not in _imported_roots(path), f"api/{path.name} imports polars"

    def test_every_service_method_returns_a_dto_or_primitive(self):
        """No public service method may be annotated as returning a DataFrame."""
        for path in _module_files("services"):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                if not isinstance(node, ast.FunctionDef) or node.name.startswith("_"):
                    continue
                returns = ast.unparse(node.returns) if node.returns else ""
                assert "DataFrame" not in returns, (
                    f"services/{path.name}::{node.name} returns a DataFrame"
                )
