#!/usr/bin/env python3
"""Train QANet."""

from __future__ import annotations

import sys

from train_extractive import main


if __name__ == "__main__":
    sys.argv[1:1] = ["--model", "qanet"]
    main()
