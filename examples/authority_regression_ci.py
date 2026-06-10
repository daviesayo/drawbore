"""Fail CI when a pipeline edit grants new reachable authority.

Run in CI with two serialized manifests (the merge base and the PR head):

    python examples/authority_regression_ci.py base.json head.json
"""
import json
import pathlib
import sys

from drawbore.config import PipelineConfig, check_no_new_authority
from drawbore.config.errors import AuthorityRegressionError


def main(old_path: str, new_path: str) -> int:
    old = PipelineConfig.model_validate(json.loads(pathlib.Path(old_path).read_text()))
    new = PipelineConfig.model_validate(json.loads(pathlib.Path(new_path).read_text()))
    try:
        check_no_new_authority(old, new)
    except AuthorityRegressionError as exc:
        print(exc)            # prints the legible certificate
        return 1
    print("Authority regression check: PASSED — no new capability granted.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1], sys.argv[2]))
