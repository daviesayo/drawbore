"""Allow ``python -m drawbore`` to run the CLI."""

from __future__ import annotations

import sys

from drawbore.cli import main

if __name__ == "__main__":
    sys.exit(main())
