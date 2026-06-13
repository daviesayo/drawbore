"""Durable, file-backed checkpoint store for cross-restart resume.

``FileCheckpointStore`` persists the full checkpoint contract to a directory so
a run that halted in one process can resume in another: step outputs, skip
marks, the run's topology fingerprint, per-step semantic seals, and per-step
trust labels. A fresh instance pointed at the same directory restores all of
it, so refuse-on-drift resume works across a process restart without re-running
completed steps — and a genuine contract change still refuses with
``resume_drift``.

Durability properties
---------------------
- **Atomic per-step writes.** Every file is written to a unique temp file and
  then ``os.replace``-d into place, which is atomic on a POSIX filesystem. A
  crash mid-write leaves at most a stray ``*.tmp-*`` file (ignored on read); a
  previously committed checkpoint is never torn.
- **Secure output reconstruction.** Outputs are stored as plain JSON field
  data — never a Python class path. ``output_of`` rebuilds the typed model
  using the live pipeline's output model for that step, supplied via
  ``bind_models``. The store never imports or evaluates a type named in
  persisted data, so a tampered checkpoint cannot trigger an arbitrary import.

Stdlib only; no new dependency.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping

from pydantic import BaseModel

from drawbore.state.checkpoint import CheckpointStore
from drawbore.state.step_seal import StepSeal
from drawbore.tools.taint import TrustLabel

#: Bumped only if the on-disk layout changes incompatibly.
_SCHEMA_VERSION = 1


class FileCheckpointStore(CheckpointStore):
    """A ``CheckpointStore`` that persists to a directory on disk.

    One subdirectory per ``run_id`` (named by a hash of the id, so any string
    is a safe directory name), holding a ``run.json`` for run-level metadata
    (the topology fingerprint) and one ``step-NNNNN.json`` per checkpointed
    step. Reads are cached in memory after the first directory load.
    """

    def __init__(self, root: str | os.PathLike[str]) -> None:
        self._root = Path(root)
        self._root.mkdir(parents=True, exist_ok=True)
        # run_id -> {"loaded": bool, "fingerprint": str|None,
        #            "steps": {step_index: record_dict}}
        self._cache: dict[str, dict[str, Any]] = {}
        # run_id -> {step_index: live output model class}, from bind_models.
        self._models: dict[str, dict[int, type[BaseModel]]] = {}

    # --- path helpers -------------------------------------------------------

    def _run_dir(self, run_id: str) -> Path:
        # Hash the run id so any string (slashes, dots, unicode) maps to a
        # single safe, collision-resistant directory name.
        digest = hashlib.sha256(run_id.encode("utf-8")).hexdigest()
        return self._root / digest

    @staticmethod
    def _step_filename(step: int) -> str:
        # Zero-padded for stable lexical ordering when a human lists the dir.
        return f"step-{step:05d}.json"

    def _atomic_write(self, path: Path, data: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(data, separators=(",", ":"), sort_keys=True)
        # Unique temp name so concurrent writers never share a temp file; the
        # ``.tmp-`` infix is what read-side scans skip.
        tmp = path.with_name(f"{path.name}.tmp-{os.getpid()}-{os.urandom(6).hex()}")
        with open(tmp, "w", encoding="utf-8") as fh:
            fh.write(payload)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)  # atomic on POSIX; overwrites any prior commit

    # --- cache / load -------------------------------------------------------

    def _run_cache(self, run_id: str) -> dict[str, Any]:
        cache = self._cache.get(run_id)
        if cache is not None and cache["loaded"]:
            return cache
        cache = {"loaded": True, "fingerprint": None, "steps": {}}
        run_dir = self._run_dir(run_id)
        if run_dir.is_dir():
            run_file = run_dir / "run.json"
            if run_file.is_file():
                meta = self._read_json(run_file)
                if meta is not None:
                    cache["fingerprint"] = meta.get("fingerprint")
            for child in run_dir.iterdir():
                name = child.name
                # Only fully committed step files; never a stray temp file.
                if not (name.startswith("step-") and name.endswith(".json")):
                    continue
                record = self._read_json(child)
                if record is None:
                    continue
                try:
                    index = int(name[len("step-"):-len(".json")])
                except ValueError:
                    continue
                cache["steps"][index] = record
        self._cache[run_id] = cache
        return cache

    @staticmethod
    def _read_json(path: Path) -> dict[str, Any] | None:
        try:
            with open(path, "r", encoding="utf-8") as fh:
                return json.load(fh)
        except (OSError, json.JSONDecodeError):
            # A torn or unreadable file is treated as absent; the committed
            # files around it remain valid.
            return None

    def _step_record(self, run_id: str, step: int) -> dict[str, Any]:
        cache = self._run_cache(run_id)
        record = cache["steps"].get(step)
        if record is None:
            record = {}
        return record

    def _put_step_record(self, run_id: str, step: int, record: dict[str, Any]) -> None:
        cache = self._run_cache(run_id)
        cache["steps"][step] = record
        self._atomic_write(
            self._run_dir(run_id) / self._step_filename(step), record
        )

    def _merge_step_record(self, run_id: str, step: int, **fields: Any) -> None:
        """Read-copy-mutate-write a step record, merging ``fields`` into it."""
        record = dict(self._step_record(run_id, step))
        record.update(fields)
        self._put_step_record(run_id, step, record)

    # --- contract: outputs --------------------------------------------------

    def step_started(self, run_id: str, step: int) -> None:
        # No durable marker: resume needs only `is_completed`/`output_of`, and a
        # step that merely started is correctly re-run on resume. Recording it
        # would add a write per step with no resume benefit.
        return None

    def step_succeeded(self, run_id: str, step: int, output: BaseModel) -> None:
        self._merge_step_record(run_id, step, completed=True, output=output.model_dump(mode="json"))

    def is_completed(self, run_id: str, step: int) -> bool:
        return bool(self._step_record(run_id, step).get("completed"))

    def output_of(self, run_id: str, step: int) -> BaseModel:
        record = self._step_record(run_id, step)
        if not record.get("completed") or "output" not in record:
            raise KeyError(
                f"step {step} of run {run_id!r} has no checkpointed output"
            )
        models = self._models.get(run_id)
        model = models.get(step) if models is not None else None
        if model is None:
            # Fail closed: we never reconstruct a type from persisted data, so
            # without the live model from `bind_models` there is nothing safe
            # to return.
            raise KeyError(
                f"no output model bound for step {step} of run {run_id!r}; "
                "call bind_models before restoring"
            )
        return model.model_validate(record["output"])

    # --- contract: skip marks ----------------------------------------------

    def step_skipped(self, run_id: str, step: int) -> None:
        self._merge_step_record(run_id, step, skipped=True)

    def is_skipped(self, run_id: str, step: int) -> bool:
        return bool(self._step_record(run_id, step).get("skipped"))

    # --- contract: topology fingerprint ------------------------------------

    def record_fingerprint(self, run_id: str, fingerprint: str) -> None:
        cache = self._run_cache(run_id)
        cache["fingerprint"] = fingerprint
        self._atomic_write(
            self._run_dir(run_id) / "run.json",
            {"run_id": run_id, "fingerprint": fingerprint,
             "schema_version": _SCHEMA_VERSION},
        )

    def fingerprint_matches(self, run_id: str, fingerprint: str) -> bool:
        stored = self._run_cache(run_id)["fingerprint"]
        return stored is None or stored == fingerprint

    # --- contract: seals ----------------------------------------------------

    def record_seal(self, run_id: str, step: int, seal: StepSeal) -> None:
        self._merge_step_record(run_id, step, seal=seal.model_dump(mode="json"))

    def seal_of(self, run_id: str, step: int) -> StepSeal | None:
        seal = self._step_record(run_id, step).get("seal")
        if seal is None:
            return None
        return StepSeal.model_validate(seal)

    # --- contract: trust labels --------------------------------------------

    def record_trust(self, run_id: str, step: int, trust: TrustLabel) -> None:
        self._merge_step_record(run_id, step, trust=trust.value)

    def trust_of(self, run_id: str, step: int) -> TrustLabel:
        value = self._step_record(run_id, step).get("trust")
        if value is None:
            return TrustLabel.UNTRUSTED
        return TrustLabel(value)

    # --- contract: approval request persistence ----------------------------

    _APPROVAL_FILENAME = "approval-request.json"

    def record_approval_request(self, run_id: str, request: dict[str, Any]) -> None:
        """Persist the JSON dict of the pending approval request to
        ``approval-request.json`` inside the run directory.

        The caller (the pipeline) serialises the typed request to a dict via
        ``model_dump(mode="json")`` before passing it here.  This store never
        imports or reconstructs a typed approval object from persisted data.
        """
        self._atomic_write(
            self._run_dir(run_id) / self._APPROVAL_FILENAME,
            request,
        )

    def approval_request_of(self, run_id: str) -> dict[str, Any] | None:
        """Read ``approval-request.json`` fresh from disk and return it as a
        plain dict, or ``None`` if absent.

        The caller (the pipeline) reconstructs the typed request via
        ``ApprovalRequest.model_validate(data)``.  This store only stores and
        retrieves JSON — it imports no type from a higher subsystem.
        """
        path = self._run_dir(run_id) / self._APPROVAL_FILENAME
        return self._read_json(path)

    def clear_approval_request(self, run_id: str) -> None:
        """Remove ``approval-request.json``; idempotent if absent."""
        try:
            os.remove(self._run_dir(run_id) / self._APPROVAL_FILENAME)
        except FileNotFoundError:
            pass

    # --- contract: model binding for secure reconstruction ------------------

    def bind_models(
        self, run_id: str, models: Mapping[int, type[BaseModel]]
    ) -> None:
        self._models[run_id] = dict(models)
