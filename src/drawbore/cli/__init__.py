"""The ``drawbore`` command-line tool.

Drawbore's thesis is that a pipeline's *shape and policy* is a reviewable
artifact and that no safety property may regress silently. This CLI puts that
thesis on the command line: its center of gravity is ``drawbore check`` — a
fail-closed gate any CI job can run over a JSON manifest, with no running
system and (for the manifest-only checks) no application code at all.

Verbs
-----
- ``check`` — validate a manifest structurally; optionally resolve it against
  live code (drift-checked), and/or diff it against a baseline manifest for
  authority widening and input-schema relaxation. Exits non-zero on any
  failure, printing the legible reason.
- ``new`` — scaffold a runnable, typed starter *package* (pipeline + test).
- ``version`` — print the installed Drawbore version.

Stdlib only (``argparse``); no new dependency. The CLI imports the config
layer lazily inside each handler so ``drawbore version`` and ``drawbore new``
stay fast and dependency-light.
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path
from typing import Any, Sequence

# Exit codes are part of the CLI contract (CI branches on them):
#   0 — every requested check passed / command succeeded
#   1 — a check failed (drift, authority widening, schema relaxation, bad manifest)
#   2 — usage error (argparse default; bad arguments, unreadable file, bad --catalog)
EXIT_OK = 0
EXIT_CHECK_FAILED = 1
EXIT_USAGE = 2


def main(argv: "Sequence[str] | None" = None) -> int:
    """Entry point. Returns a process exit code (see module docstring)."""
    parser = _build_parser()
    args = parser.parse_args(argv)
    handler = getattr(args, "_handler", None)
    if handler is None:  # no subcommand given
        parser.print_help(sys.stderr)
        return EXIT_USAGE
    return handler(args)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="drawbore",
        description="Constrained, composable agents that hold by construction.",
    )
    sub = parser.add_subparsers(dest="command", metavar="<command>")

    check = sub.add_parser(
        "check",
        help="Validate a pipeline manifest; optionally resolve and ratchet it.",
        description=(
            "Fail-closed gate over a JSON pipeline manifest. With no options it "
            "validates the manifest's structure. --catalog resolves it against "
            "live agent code and fails on declaration drift. --against diffs it "
            "against a baseline manifest and fails on any authority widening or "
            "input-schema relaxation. Combine freely; the command exits non-zero "
            "if any requested check fails."
        ),
    )
    check.add_argument("manifest", help="Path to the candidate pipeline manifest (JSON).")
    check.add_argument(
        "--catalog",
        metavar="MODULE:ATTR",
        help=(
            "Import path to an AgentCatalog (or Mapping[str, Agent]) to resolve "
            "the manifest's agent refs against live code, drift-checked."
        ),
    )
    check.add_argument(
        "--registry",
        metavar="MODULE:ATTR",
        help=(
            "Import path to a ToolRegistry for the resolution check. Defaults to "
            "the process-global registry (tools registered as an import side "
            "effect of --catalog's module)."
        ),
    )
    check.add_argument(
        "--against",
        metavar="BASELINE",
        help=(
            "Path to a baseline manifest. The candidate is rejected if it grants "
            "new capability authority or relaxes any agent's input schema."
        ),
    )
    check.set_defaults(_handler=_cmd_check)

    new = sub.add_parser(
        "new",
        help="Scaffold a runnable, typed starter pipeline package.",
        description=(
            "Write a minimal, runnable Drawbore pipeline package to <name>/ "
            "(pipeline + test + README), mirroring the layout of the project's "
            "own examples/."
        ),
    )
    new.add_argument("name", help="Package + pipeline name (a valid Python identifier).")
    new.add_argument(
        "--dir",
        default=".",
        metavar="DIR",
        help="Directory to create the package in (default: current directory).",
    )
    new.set_defaults(_handler=_cmd_new)

    version = sub.add_parser("version", help="Print the installed Drawbore version.")
    version.set_defaults(_handler=_cmd_version)

    return parser


# ---------------------------------------------------------------------------
# check
# ---------------------------------------------------------------------------

def _cmd_check(args: argparse.Namespace) -> int:
    # Lazy imports: keep `version`/`new` free of the config layer.
    from drawbore.config import (
        ConfigResolutionError,
        PipelineConfig,
        check_no_new_authority,
        check_no_schema_relaxation,
        from_json,
    )
    from drawbore.config.errors import AuthorityRegressionError, SchemaRelaxationError

    text = _read_text(args.manifest)
    if text is None:
        return EXIT_USAGE

    # 1. Structural validation (always). Unknown fields and a bad schema_version
    #    fail closed here, before any code is imported.
    candidate = _parse_config(text, args.manifest, PipelineConfig)
    if candidate is None:
        return EXIT_CHECK_FAILED
    n_agents = len(candidate.agents)
    n_steps = len(candidate.steps)
    print(
        f"manifest OK — {n_agents} agent(s), {n_steps} step(s) "
        f"(pipeline '{candidate.pipeline.name}' v{candidate.pipeline.version})."
    )

    # 2. Resolution against live code (optional, drift-checked).
    if args.catalog is not None:
        catalog = _load_attr(args.catalog)
        if catalog is None:
            return EXIT_USAGE
        registry = None
        if args.registry is not None:
            registry = _load_attr(args.registry)
            if registry is None:
                return EXIT_USAGE
        try:
            from_json(text, agents=catalog, registry=registry)
        except ConfigResolutionError as exc:
            print(f"resolution FAILED — {exc}", file=sys.stderr)
            return EXIT_CHECK_FAILED
        print("resolution OK — manifest resolves against live code with no drift.")

    # 3. Safety ratchet against a baseline (optional, manifest-only).
    if args.against is not None:
        baseline_text = _read_text(args.against)
        if baseline_text is None:
            return EXIT_USAGE
        baseline = _parse_config(baseline_text, args.against, PipelineConfig)
        if baseline is None:
            return EXIT_CHECK_FAILED
        failed = False
        try:
            check_no_new_authority(baseline, candidate)
            print("authority OK — no new capability authority granted.")
        except AuthorityRegressionError as exc:
            print(str(exc), file=sys.stderr)
            failed = True
        try:
            check_no_schema_relaxation(baseline, candidate)
            print("input-schema OK — no agent input schema relaxed.")
        except SchemaRelaxationError as exc:
            print(str(exc), file=sys.stderr)
            failed = True
        if failed:
            return EXIT_CHECK_FAILED

    return EXIT_OK


def _parse_config(text: str, source: str, config_cls: Any) -> Any:
    """Parse JSON text into a ``PipelineConfig`` (fail-closed), or print why not."""
    from pydantic import ValidationError

    try:
        raw = json.loads(text)
    except json.JSONDecodeError as exc:
        print(f"manifest FAILED — {source} is not valid JSON: {exc}", file=sys.stderr)
        return None
    try:
        return config_cls.model_validate(raw)
    except ValidationError as exc:
        print(f"manifest FAILED — {source} is not a valid manifest: {exc}", file=sys.stderr)
        return None


# ---------------------------------------------------------------------------
# new — a runnable, typed starter package
# ---------------------------------------------------------------------------

_TPL_INIT = '''\
"""The {name} pipeline package."""
'''

_TPL_PIPELINE = '''\
"""The {name} pipeline: typed agents composed into a schema-checked pipeline.

Every step is schema-checked at both edges; a wrong shape halts the run with a
readable reason. Replace the body of ``normalize`` with your own logic and add
more agents with ``pipeline.add(...)``.
"""

from pydantic import BaseModel

from drawbore import Pipeline, agent
from drawbore.config import AgentCatalog


class Input(BaseModel):
    cents: int


class Output(BaseModel):
    dollars: float


@agent(name="normalize", input=Input, output=Output)
async def normalize(value: Input) -> Output:
    return Output(dollars=value.cents / 100)


pipeline = Pipeline(name="{name}", version="0.1.0")
pipeline.add(normalize)

# A catalog gives every agent a stable symbolic ref, so the pipeline can be
# exported to (and checked against) a JSON manifest with `drawbore check`.
catalog = AgentCatalog()
catalog.register("{name}.pipeline:normalize", normalize)
'''

_TPL_TEST = '''\
"""Tests for the {name} pipeline.

Uses ``asyncio.run`` inside sync tests so it passes under a plain ``pytest``
invocation, with no asyncio-mode configuration required.
"""

import asyncio

from drawbore.config import from_config, to_config

from {name}.pipeline import Input, Output, catalog, pipeline


def test_pipeline_runs():
    result = asyncio.run(pipeline.run(Input(cents=2500)))
    assert result.status == "completed"
    assert result.outputs["normalize"] == Output(dollars=25.0)


def test_manifest_round_trips():
    # The pipeline's shape + policy round-trips through a JSON manifest, and the
    # rebuilt pipeline re-exports to the identical config. This is the contract
    # `drawbore check` enforces in CI.
    config = to_config(pipeline, agents=catalog)
    rebuilt = from_config(config, agents=catalog)
    assert to_config(rebuilt, agents=catalog) == config
'''

_TPL_README = '''\
# {name}

A starter Drawbore pipeline: typed agents composed into a schema-checked
pipeline. Every step is validated at both edges; a wrong shape halts the run.

## Run it

```bash
python -c "import asyncio; from {name}.pipeline import pipeline, Input; \\
print(asyncio.run(pipeline.run(Input(cents=2500))).outputs)"
```

## Test it

```bash
pytest {name}/
```

## Check it in CI

Export a reviewable JSON manifest, then gate it with the Drawbore CLI:

```bash
python -c "from {name}.pipeline import pipeline, catalog; from drawbore.config import to_json; \\
open('{name}/manifest.json', 'w').write(to_json(pipeline, agents=catalog))"

drawbore check {name}/manifest.json --catalog {name}.pipeline:catalog
```

Add `--against BASELINE.json` to additionally fail the check if a change grants
new tool authority or relaxes an agent's input schema.
'''


def _cmd_new(args: argparse.Namespace) -> int:
    name = args.name
    if not name.isidentifier():
        print(
            f"'{name}' is not a valid Python identifier; pick a name usable as a "
            "package and pipeline name (letters, digits, underscores; no leading digit).",
            file=sys.stderr,
        )
        return EXIT_USAGE
    pkg = Path(args.dir) / name
    if pkg.exists():
        print(f"refusing to overwrite existing path {pkg}", file=sys.stderr)
        return EXIT_USAGE

    files = {
        "__init__.py": _TPL_INIT,
        "pipeline.py": _TPL_PIPELINE,
        f"test_{name}.py": _TPL_TEST,
        "README.md": _TPL_README,
    }
    try:
        pkg.mkdir(parents=True)
        for filename, template in files.items():
            (pkg / filename).write_text(template.format(name=name), encoding="utf-8")
    except OSError as exc:
        print(f"could not write package {pkg}: {exc}", file=sys.stderr)
        return EXIT_USAGE

    print(f"created package {pkg}/")
    for filename in files:
        print(f"  {name}/{filename}")
    print()
    print(f"run the tests with:  pytest {pkg}/")
    return EXIT_OK


# ---------------------------------------------------------------------------
# version
# ---------------------------------------------------------------------------

def _cmd_version(_args: argparse.Namespace) -> int:
    from importlib.metadata import PackageNotFoundError, version

    try:
        print(f"drawbore {version('drawbore')}")
    except PackageNotFoundError:
        print("drawbore (version unknown — not installed as a distribution)")
    return EXIT_OK


# ---------------------------------------------------------------------------
# shared helpers
# ---------------------------------------------------------------------------

def _read_text(path: str) -> "str | None":
    try:
        return Path(path).read_text(encoding="utf-8")
    except OSError as exc:
        print(f"could not read {path}: {exc}", file=sys.stderr)
        return None


def _load_attr(spec: str) -> Any:
    """Resolve a ``module:attr`` import path to its object, or print why not.

    Returns the object on success, or ``None`` after printing a usage error.
    """
    if ":" not in spec:
        print(
            f"--catalog/--registry value '{spec}' must be 'module:attr' "
            "(e.g. 'myapp.pipelines:catalog').",
            file=sys.stderr,
        )
        return None
    module_name, _, attr = spec.partition(":")
    try:
        module = importlib.import_module(module_name)
    except Exception as exc:  # import side effects can raise anything
        print(f"could not import module '{module_name}': {exc}", file=sys.stderr)
        return None
    try:
        return getattr(module, attr)
    except AttributeError:
        print(f"module '{module_name}' has no attribute '{attr}'", file=sys.stderr)
        return None
