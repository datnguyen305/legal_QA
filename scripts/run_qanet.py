#!/usr/bin/env python3
"""Run QANet inference."""

from __future__ import annotations

import sys

from run_extractive import main


if __name__ == "__main__":
    sys.argv[1:1] = ["--expected-model", "qanet"]
    main()
