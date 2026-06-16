#!/usr/bin/env python3
"""Train Cross-Passage Answer Verification."""

from __future__ import annotations

import sys

from train_extractive import main


if __name__ == "__main__":
    sys.argv[1:1] = ["--model", "cross_passage"]
    main()
