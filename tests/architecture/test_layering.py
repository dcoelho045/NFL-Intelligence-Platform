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
        "xgboost",
        "torch",
        "tensorflow",
        "mlflow",
        "dvc",
    }
)

# Milestone 6 adds the ML libraries. Same quarantine reasoning as every SDK
# before them: "which code can fit a model" stays answerable by naming one
# package. xgboost stays forbidden -- it is redundant with LightGBM here, and
# torch/tensorflow stay forbidden because this milestone decided against a deep
# learning model rather than merely not getting to one.
ML_PACKAGES = frozenset({"sklearn", "lightgbm"})
ML_ADAPTER_PACKAGE = "ml/models"

# Training, evaluation and dataset assembly are offline-only. Serving code may
# load an artifact and score it; it must never be able to fit one. Enforced as
# an import rule because the alternative -- a convention -- is one convenience
# import away from an API process that can retrain itself on a request.
TRAINING_ONLY_MODULES = (
    "nfl_platform.ml.training",
    "nfl_platform.ml.evaluation",
    "nfl_platform.ml.datasets",
    "nfl_platform.ml.cli",
)
SERVING_PACKAGES = ("services", "api", "llm", "mcp_server", "ml/predict", "ml/registry")

# Milestone 3 introduces exactly one LLM SDK. Milestone 4 uses it for a second
# purpose (embeddings) from a layer that must not import upward into llm/, so the
# rule is now "vendor SDK calls live in an enumerated set of provider packages"
# rather than "one package". The property that matters is unchanged: every file
# that can send text to a vendor can be listed.
PROVIDER_ONLY_PACKAGES = frozenset({"openai"})
PROVIDER_PACKAGES = ("llm/providers", "retrieval/providers")

# Milestone 4 adds web retrieval. The HTTP client is quarantined the same way:
# only the search adapters may speak to the network, so "which code can make an
# outbound request" stays answerable by listing two files.
HTTP_CLIENT_PACKAGES = frozenset({"httpx", "requests", "urllib3", "aiohttp"})
HTTP_ALLOWED_MODULES = (
    "retrieval/web/tavily.py",
    "retrieval/web/brave.py",
)

# Milestone 5 adds the MCP SDK. Same quarantine reasoning as the LLM SDK: the
# protocol library belongs to the adapter that speaks the protocol, so "which
# code depends on MCP" stays answerable by naming one package.
MCP_SDK_PACKAGES = frozenset({"mcp"})
MCP_ADAPTER_PACKAGE = "mcp_server"


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

    def test_detects_ml_library_imports(self, tmp_path: Path):
        sample = tmp_path / "sample.py"
        sample.write_text("import lightgbm as lgb\n", encoding="utf-8")
        assert _imported_roots(sample) & ML_PACKAGES == {"lightgbm"}

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
    """The model SDK lives in enumerated packages; nothing else may call a model."""

    @pytest.mark.parametrize("path", _module_files(""), ids=lambda p: p.name)
    def test_only_the_provider_packages_import_an_llm_sdk(self, path: Path):
        offenders = _imported_roots(path) & PROVIDER_ONLY_PACKAGES
        if any(package in path.as_posix() for package in PROVIDER_PACKAGES):
            return
        assert not offenders, (
            f"{path.relative_to(SRC).as_posix()} imports {sorted(offenders)}; "
            f"LLM SDK calls belong in {' or '.join(PROVIDER_PACKAGES)}/"
        )

    def test_the_provider_packages_actually_exist(self):
        """Guard against the rule passing vacuously if a package is renamed."""
        for package in PROVIDER_PACKAGES:
            assert (SRC / package).is_dir(), package

    @pytest.mark.parametrize("path", _module_files("analytics"), ids=lambda p: p.name)
    def test_analytics_never_imports_the_llm_layer(self, path: Path):
        offenders = {m for m in _imported_modules(path) if m.startswith("nfl_platform.llm")}
        assert not offenders, f"analytics/{path.name} imports the LLM layer"

    @pytest.mark.parametrize("path", _module_files("services"), ids=lambda p: p.name)
    def test_services_never_import_the_llm_layer(self, path: Path):
        """The deterministic core must stay usable with no model present."""
        offenders = {m for m in _imported_modules(path) if m.startswith("nfl_platform.llm")}
        assert not offenders, f"services/{path.name} imports the LLM layer"


class TestHttpClientIsQuarantined:
    """Milestone 4: exactly two files may make an outbound web request."""

    @pytest.mark.parametrize("path", _module_files(""), ids=lambda p: p.name)
    def test_only_the_search_adapters_import_an_http_client(self, path: Path):
        offenders = _imported_roots(path) & HTTP_CLIENT_PACKAGES
        relative = path.relative_to(SRC).as_posix()
        if relative in HTTP_ALLOWED_MODULES:
            return
        assert not offenders, (
            f"{relative} imports {sorted(offenders)}; outbound HTTP belongs in "
            f"{', '.join(HTTP_ALLOWED_MODULES)}"
        )

    def test_the_allowed_http_modules_actually_exist(self):
        """Guard against the rule passing vacuously if a file is renamed."""
        for relative in HTTP_ALLOWED_MODULES:
            assert (SRC / relative).is_file(), relative

    @pytest.mark.parametrize("path", _module_files("retrieval"), ids=lambda p: p.name)
    def test_retrieval_never_imports_an_llm_sdk(self, path: Path):
        """Embeddings go through a Protocol; the SDK stays in retrieval/providers/."""
        offenders = _imported_roots(path) & PROVIDER_ONLY_PACKAGES
        if "retrieval/providers" in path.as_posix():
            return
        assert not offenders, f"retrieval/{path.name} imports {sorted(offenders)}"


class TestMcpSdkIsQuarantined:
    """Milestone 5: only the MCP adapter may import the MCP SDK."""

    @pytest.mark.parametrize("path", _module_files(""), ids=lambda p: p.name)
    def test_only_the_mcp_adapter_imports_the_mcp_sdk(self, path: Path):
        offenders = _imported_roots(path) & MCP_SDK_PACKAGES
        if MCP_ADAPTER_PACKAGE in path.as_posix():
            return
        assert not offenders, (
            f"{path.relative_to(SRC).as_posix()} imports {sorted(offenders)}; "
            f"MCP SDK calls belong in {MCP_ADAPTER_PACKAGE}/"
        )

    def test_the_mcp_adapter_package_exists(self):
        """Guard against the rule passing vacuously if the package is renamed."""
        assert (SRC / MCP_ADAPTER_PACKAGE).is_dir()

    def test_the_mcp_adapter_actually_uses_the_sdk(self):
        """A quarantine around a package that never imports the SDK proves nothing."""
        importers = [
            path
            for path in _module_files(MCP_ADAPTER_PACKAGE)
            if _imported_roots(path) & MCP_SDK_PACKAGES
        ]
        assert importers


class TestMlLibrariesAreQuarantined:
    """Milestone 6: only the model package may fit a model."""

    @pytest.mark.parametrize("path", _module_files(""), ids=lambda p: p.name)
    def test_only_the_model_package_imports_an_ml_library(self, path: Path):
        offenders = _imported_roots(path) & ML_PACKAGES
        if ML_ADAPTER_PACKAGE in path.as_posix():
            return
        assert not offenders, (
            f"{path.relative_to(SRC).as_posix()} imports {sorted(offenders)}; "
            f"estimator code belongs in {ML_ADAPTER_PACKAGE}/"
        )

    def test_the_model_package_exists(self):
        assert (SRC / ML_ADAPTER_PACKAGE).is_dir()

    def test_the_model_package_actually_uses_the_libraries(self):
        """A quarantine around a package that never imports them proves nothing."""
        importers = [
            path
            for path in _module_files(ML_ADAPTER_PACKAGE)
            if _imported_roots(path) & ML_PACKAGES
        ]
        assert importers

    def test_exactly_the_agreed_ml_libraries_are_declared(self):
        pyproject = tomllib.loads((SRC.parents[1] / "pyproject.toml").read_text(encoding="utf-8"))
        roots = {
            re.split(r"[<>=!~\[; ]", entry, maxsplit=1)[0].lower()
            for entry in pyproject["project"]["dependencies"]
        }
        assert "scikit-learn" in roots
        assert "lightgbm" in roots
        assert not roots & {"xgboost", "torch", "tensorflow", "mlflow", "dvc"}

    @pytest.mark.parametrize("path", _module_files("ml"), ids=lambda p: p.name)
    def test_the_ml_layer_never_imports_a_model_provider(self, path: Path):
        """The prediction layer contains no LLM. That is the milestone's whole point."""
        offenders = _imported_roots(path) & PROVIDER_ONLY_PACKAGES
        assert not offenders, f"ml/{path.name} imports {sorted(offenders)}"

    @pytest.mark.parametrize("path", _module_files("ml"), ids=lambda p: p.name)
    def test_the_ml_layer_never_imports_the_llm_layer(self, path: Path):
        offenders = {m for m in _imported_modules(path) if m.startswith("nfl_platform.llm")}
        assert not offenders, f"ml/{path.name} imports the LLM layer"


class TestTrainingIsSeparatedFromServing:
    """Milestone 6: the thing that scores a model cannot fit one."""

    @pytest.mark.parametrize("package", SERVING_PACKAGES, ids=lambda p: p.replace("/", "."))
    def test_serving_code_never_imports_a_training_module(self, package: str):
        offenders: dict[str, list[str]] = {}
        for path in _module_files(package):
            found = sorted(
                module
                for module in _imported_modules(path)
                if module.startswith(TRAINING_ONLY_MODULES)
            )
            if found:
                offenders[path.relative_to(SRC).as_posix()] = found
        assert not offenders, f"serving code imports training modules: {offenders}"

    def test_the_training_modules_exist(self):
        """Guard against the rule passing vacuously if a package is renamed."""
        for module in TRAINING_ONLY_MODULES:
            relative = module.removeprefix("nfl_platform.").replace(".", "/")
            path = SRC / relative
            assert path.is_dir() or path.with_suffix(".py").is_file(), module

    def test_the_predictors_expose_no_fit_method(self):
        """`predict` is the whole serving contract; `fit` would widen it silently."""
        for path in _module_files("ml/predict"):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            names = {node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)}
            assert "fit" not in names, f"ml/predict/{path.name} defines fit()"


class TestLayerDependencyDirection:
    @pytest.mark.parametrize("path", _module_files("ml"), ids=lambda p: p.name)
    def test_ml_does_not_depend_on_anything_above_it(self, path: Path):
        """ml sits beside analytics and retrieval: services compose it, it composes nothing."""
        offenders = {
            module
            for module in _imported_modules(path)
            if module.startswith(
                ("nfl_platform.services", "nfl_platform.api", "nfl_platform.mcp_server")
            )
        }
        assert not offenders, f"ml/{path.name} imports upward: {sorted(offenders)}"

    @pytest.mark.parametrize("path", _module_files("ml"), ids=lambda p: p.name)
    def test_ml_never_imports_the_web_framework_or_an_http_client(self, path: Path):
        offenders = _imported_roots(path) & (
            {"fastapi", "starlette", "uvicorn"} | HTTP_CLIENT_PACKAGES | MCP_SDK_PACKAGES
        )
        assert not offenders, f"ml/{path.name} imports {sorted(offenders)}"

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
        """Routers must not call the data layer, analytics, or retrieval directly."""
        modules = _imported_modules(path)
        offenders = {
            module
            for module in modules
            if module.startswith(
                ("nfl_platform.analytics", "nfl_platform.data", "nfl_platform.retrieval")
            )
        }
        assert not offenders, f"api/{path.name} bypasses the service layer: {sorted(offenders)}"

    @pytest.mark.parametrize("path", _module_files("llm"), ids=lambda p: p.name)
    def test_llm_does_not_reach_past_the_service_layer(self, path: Path):
        """Tools must call services, never recompute from frames or retrieve directly.

        The retrieval clause matters as much as the analytics one: a tool that
        reached into ``retrieval/`` would own search policy, and search policy
        would then be untestable without a model.
        """
        modules = _imported_modules(path)
        offenders = {
            module
            for module in modules
            if module.startswith(
                ("nfl_platform.analytics", "nfl_platform.data", "nfl_platform.retrieval")
            )
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

    @pytest.mark.parametrize("path", _module_files("retrieval"), ids=lambda p: p.name)
    def test_retrieval_does_not_depend_on_anything_above_it(self, path: Path):
        """retrieval sits beside analytics: services compose it, it composes nothing."""
        modules = _imported_modules(path)
        offenders = {
            module
            for module in modules
            if module.startswith(("nfl_platform.services", "nfl_platform.api", "nfl_platform.llm"))
        }
        assert not offenders, f"retrieval/{path.name} imports upward: {sorted(offenders)}"

    @pytest.mark.parametrize("path", _module_files("retrieval"), ids=lambda p: p.name)
    def test_retrieval_never_imports_the_web_framework(self, path: Path):
        offenders = _imported_roots(path) & {"fastapi", "starlette", "uvicorn"}
        assert not offenders, f"retrieval/{path.name} imports {sorted(offenders)}"

    @pytest.mark.parametrize("path", _module_files(MCP_ADAPTER_PACKAGE), ids=lambda p: p.name)
    def test_mcp_does_not_reach_past_the_service_layer(self, path: Path):
        """MCP is an interface, not a second implementation of the analytics."""
        modules = _imported_modules(path)
        offenders = {
            module
            for module in modules
            if module.startswith(
                ("nfl_platform.analytics", "nfl_platform.data", "nfl_platform.retrieval")
            )
        }
        assert not offenders, (
            f"{MCP_ADAPTER_PACKAGE}/{path.name} bypasses the service layer: {sorted(offenders)}"
        )

    @pytest.mark.parametrize("path", _module_files(MCP_ADAPTER_PACKAGE), ids=lambda p: p.name)
    def test_mcp_does_not_depend_on_the_other_adapters(self, path: Path):
        """api, llm and mcp_server are siblings over one container, not a stack.

        An import in either direction would make one adapter's design decisions
        binding on the others, which is exactly what the shared neutral modules
        (schemas/fields.py, tool_errors.py) exist to avoid.
        """
        offenders = {
            module
            for module in _imported_modules(path)
            if module.startswith(("nfl_platform.api", "nfl_platform.llm"))
        }
        assert not offenders, (
            f"{MCP_ADAPTER_PACKAGE}/{path.name} imports a sibling adapter: {sorted(offenders)}"
        )

    @pytest.mark.parametrize("path", _module_files(MCP_ADAPTER_PACKAGE), ids=lambda p: p.name)
    def test_mcp_never_imports_polars_or_the_web_framework(self, path: Path):
        offenders = _imported_roots(path) & {"polars", "fastapi", "starlette", "uvicorn"}
        assert not offenders, f"{MCP_ADAPTER_PACKAGE}/{path.name} imports {sorted(offenders)}"

    @pytest.mark.parametrize("path", _module_files("llm"), ids=lambda p: p.name)
    def test_llm_does_not_depend_on_the_mcp_adapter(self, path: Path):
        """The agent calls services in-process; routing it through MCP is not this milestone."""
        offenders = {
            module
            for module in _imported_modules(path)
            if module.startswith(f"nfl_platform.{MCP_ADAPTER_PACKAGE}")
        }
        assert not offenders, f"llm/{path.name} imports the MCP adapter: {sorted(offenders)}"


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

    def test_every_retrieval_method_returns_a_typed_result(self):
        """The vector index is Parquet-backed; the frame must not escape the module."""
        for path in _module_files("retrieval"):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                if not isinstance(node, ast.FunctionDef) or node.name.startswith("_"):
                    continue
                returns = ast.unparse(node.returns) if node.returns else ""
                assert "DataFrame" not in returns, (
                    f"retrieval/{path.name}::{node.name} returns a DataFrame"
                )
